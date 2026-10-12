//! `OpenAI` Completions conversion: reasoning-details signatures, messages, and tools.

use serde_json::{json, Map, Value};

use crate::models::{calculate_cost, CostOverrides};
use crate::providers::openai_completions::{decode_reasoning_details, ResolvedCompat};
use crate::providers::transform_messages::transform_messages_with_normalizer;
use crate::types::{
    AssistantContent, Context, Message, MessageExt, Model, ModelInput, StopReason, TextContent,
    ThinkingContent, Tool, ToolCall, Usage, UsageCost, UserMessageContent, UserOrToolContent,
};
use crate::utils_inner::sanitize_unicode::sanitize_surrogates;

/// Strip prior-turn reasoning replay from the request messages (issue #9).
///
/// The boundary is the last user message: assistant messages before it belong
/// to completed turns, so their thinking replay (the `reasoning` /
/// `reasoning_content` field, `reasoning_details`, and tool-call thought
/// signatures) is removed. Assistant messages after the boundary — the current
/// tool-call chain — keep their reasoning so mid-turn reasoning keeps
/// round-tripping. A request with no user message keeps everything (the whole
/// context is then the current chain).
///
/// Must run BEFORE [`transform_messages_with_normalizer`]: the transform
/// converts cross-model unsigned thinking into visible text, where a strip
/// could no longer find it.
fn strip_prior_turn_reasoning(messages: &[Message]) -> Vec<Message> {
    let boundary = messages
        .iter()
        .rposition(|message| matches!(message, Message::User(_)));
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| match (message, boundary) {
            (Message::Assistant(assistant), Some(boundary)) if index < boundary => {
                let mut stripped = assistant.clone();
                stripped
                    .content
                    .retain(|block| !matches!(block, AssistantContent::Thinking(_)));
                for block in &mut stripped.content {
                    if let AssistantContent::ToolCall(call) = block {
                        call.thought_signature = None;
                    }
                }
                Message::Assistant(stripped)
            }
            _ => message.clone(),
        })
        .collect()
}

/// Convert a conversation into Chat Completions `messages` params, including tool-result bridging
/// and image replay.
// Long by design: mirrors the provider's stream shape.
#[allow(clippy::too_many_lines)]
pub fn convert_messages(model: &Model, context: &Context, compat: &ResolvedCompat) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();

    // Prior-turn reasoning strip (issue #9) runs BEFORE the transform: the
    // transform folds cross-model unsigned thinking into visible text, where
    // a post-transform strip could no longer remove it.
    let stripped_messages;
    let messages: &[Message] = if compat.strip_prior_turn_reasoning {
        stripped_messages = strip_prior_turn_reasoning(&context.messages);
        &stripped_messages
    } else {
        &context.messages
    };

    let transformed = transform_messages_with_normalizer(messages, model, &|id, model, _| {
        Some(normalize_tool_call_id(id, model))
    });

    if let Some(system_prompt) = &context.system_prompt {
        let use_developer_role = model.reasoning && compat.supports_developer_role;
        let role = if use_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(json!({
            "role": role,
            "content": sanitize_surrogates(system_prompt),
        }));
    }

    let mut last_role: Option<&'static str> = None;
    let mut index = 0usize;
    while index < transformed.len() {
        let msg = &transformed[index];
        // Some providers don't allow user messages directly after tool results.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(msg, Message::User(_))
        {
            params.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
        }

        match msg {
            Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => params.push(json!({
                    "role": "user",
                    "content": sanitize_surrogates(text),
                })),
                UserMessageContent::Blocks(blocks) => {
                    let content_blocks: Vec<Value> = blocks
                        .iter()
                        .map(|item| match crate::types::user_block_payload(item) {
                            crate::types::UserBlockPayload::Text(text) => json!({
                                "type": "text",
                                "text": sanitize_surrogates(text),
                            }),
                            crate::types::UserBlockPayload::Image { data, mime_type } => json!({
                                "type": "image_url",
                                "image_url": { "url": format!("data:{mime_type};base64,{data}") },
                            }),
                            crate::types::UserBlockPayload::Opaque(json) => json!({
                                "type": "text",
                                "text": sanitize_surrogates(&json),
                            }),
                        })
                        .collect();
                    if content_blocks.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(json!({
                        "role": "user",
                        "content": content_blocks,
                    }));
                }
            },
            Message::Assistant(assistant) => {
                let mut assistant_msg = Map::new();
                assistant_msg.insert("role".into(), json!("assistant"));
                // Some providers don't accept null content; use empty string instead.
                assistant_msg.insert(
                    "content".into(),
                    if compat.requires_assistant_after_tool_result {
                        json!("")
                    } else {
                        Value::Null
                    },
                );

                let text_blocks: Vec<&TextContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => Some(text),
                        _ => None,
                    })
                    .collect();
                let assistant_text = text_blocks
                    .iter()
                    .map(|block| sanitize_surrogates(&block.text))
                    .collect::<String>();

                let replay_reasoning_details: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking) => {
                            decode_reasoning_details(thinking.thinking_signature.as_deref())
                        }
                        _ => None,
                    })
                    .flatten()
                    .collect();
                if !replay_reasoning_details.is_empty() {
                    assistant_msg
                        .insert("reasoning_details".into(), json!(replay_reasoning_details));
                }

                let non_empty_thinking_blocks: Vec<&ThinkingContent> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking)
                            if decode_reasoning_details(thinking.thinking_signature.as_deref())
                                .is_none()
                                && !thinking.thinking.trim().is_empty() =>
                        {
                            Some(thinking)
                        }
                        _ => None,
                    })
                    .collect();

                if !non_empty_thinking_blocks.is_empty() {
                    if compat.requires_thinking_as_text {
                        let thinking_text = non_empty_thinking_blocks
                            .iter()
                            .map(|block| sanitize_surrogates(&block.thinking))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        assistant_msg.insert(
                            "content".into(),
                            json!([{
                                "type": "text",
                                "text": thinking_text,
                            }]),
                        );
                        for block in &text_blocks {
                            assistant_msg["content"]
                                .as_array_mut()
                                .expect("content is an array")
                                .push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&block.text),
                                }));
                        }
                    } else {
                        // Always send assistant text as a plain string.
                        if !assistant_text.is_empty() {
                            assistant_msg.insert("content".into(), json!(assistant_text));
                        }

                        let reasoning_text = non_empty_thinking_blocks
                            .iter()
                            .map(|block| sanitize_surrogates(&block.thinking))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let reasoning_field =
                            if compat.requires_reasoning_content_on_assistant_messages {
                                Some("reasoning_content")
                            } else {
                                non_empty_thinking_blocks[0].thinking_signature.as_deref()
                            };
                        match reasoning_field {
                            Some(field) => {
                                assistant_msg.insert(field.to_string(), json!(reasoning_text));
                            }
                            None => {
                                assistant_msg.insert(
                                    "content".into(),
                                    if assistant_text.is_empty() {
                                        json!(reasoning_text)
                                    } else {
                                        json!(format!("{reasoning_text}\n\n{assistant_text}"))
                                    },
                                );
                            }
                        }
                    }
                } else if !assistant_text.is_empty() {
                    assistant_msg.insert("content".into(), json!(assistant_text));
                }

                let tool_calls: Vec<&ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    assistant_msg.insert(
                        "tool_calls".into(),
                        json!(tool_calls
                            .iter()
                            .map(|tool_call| json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": serde_json::Value::Object(tool_call.arguments.clone()).to_string(),
                                },
                            }))
                            .collect::<Vec<_>>()),
                    );
                    let reasoning_details: Vec<Value> = tool_calls
                        .iter()
                        .filter_map(|tool_call| {
                            tool_call
                                .thought_signature
                                .as_ref()
                                .and_then(|signature| serde_json::from_str(signature).ok())
                        })
                        .collect();
                    if !reasoning_details.is_empty() && replay_reasoning_details.is_empty() {
                        assistant_msg.insert("reasoning_details".into(), json!(reasoning_details));
                    }
                }
                if compat.requires_reasoning_content_on_assistant_messages
                    && model.reasoning
                    && !assistant_msg.contains_key("reasoning_content")
                {
                    assistant_msg.insert("reasoning_content".into(), json!(""));
                }
                if !replay_reasoning_details.is_empty()
                    && assistant_msg.get("content") == Some(&Value::Null)
                    && !assistant_msg.contains_key("tool_calls")
                {
                    assistant_msg.insert("content".into(), json!(""));
                }
                // Skip assistant messages that have no content and no tool calls.
                let content_value = assistant_msg.get("content");
                let has_content = match content_value {
                    Some(Value::Null) | None => false,
                    Some(Value::String(text)) => !text.is_empty(),
                    Some(Value::Array(array)) => !array.is_empty(),
                    Some(_) => true,
                };
                if !has_content
                    && !assistant_msg.contains_key("tool_calls")
                    && replay_reasoning_details.is_empty()
                {
                    index += 1;
                    continue;
                }
                params.push(Value::Object(assistant_msg));
            }
            Message::ToolResult(_) => {
                let mut image_blocks: Vec<Value> = Vec::new();
                let mut j = index;
                while j < transformed.len() {
                    let Message::ToolResult(tool_msg) = &transformed[j] else {
                        break;
                    };

                    let text_result = tool_msg
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            UserOrToolContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let has_images = tool_msg
                        .content
                        .iter()
                        .any(|block| matches!(block, UserOrToolContent::Image(_)));

                    let has_text = !text_result.is_empty();
                    let mut tool_result_msg = Map::new();
                    tool_result_msg.insert("role".into(), json!("tool"));
                    tool_result_msg.insert(
                        "content".into(),
                        json!(sanitize_surrogates(if has_text {
                            text_result.as_str()
                        } else if has_images {
                            "(see attached image)"
                        } else {
                            ""
                        })),
                    );
                    tool_result_msg.insert("tool_call_id".into(), json!(tool_msg.tool_call_id));
                    if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                        tool_result_msg.insert("name".into(), json!(tool_msg.tool_name));
                    }
                    params.push(Value::Object(tool_result_msg));

                    if has_images
                        && model
                            .input
                            .iter()
                            .any(|mode| matches!(mode, ModelInput::Image))
                    {
                        for block in &tool_msg.content {
                            if let UserOrToolContent::Image(image) = block {
                                image_blocks.push(json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
                                }));
                            }
                        }
                    }
                    j += 1;
                }

                index = j;
                if image_blocks.is_empty() {
                    last_role = Some("toolResult");
                } else {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({
                            "role": "assistant",
                            "content": "I have processed the tool results.",
                        }));
                    }
                    let mut content_items = vec![json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:",
                    })];
                    content_items.extend(image_blocks);
                    params.push(json!({
                        "role": "user",
                        "content": content_items,
                    }));
                    last_role = Some("user");
                }
                continue;
            }
        }

        last_role = Some(msg.role());
        index += 1;
    }

    params
}

/// Normalize a tool call ID for providers that reject long/pipe-separated IDs.
pub fn normalize_tool_call_id(id: &str, model: &Model) -> String {
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or("");
        call_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(40)
            .collect()
    } else if model.provider == "openai" {
        id.chars().take(40).collect()
    } else {
        id.to_string()
    }
}

pub fn convert_tools(tools: &[Tool], compat: &ResolvedCompat) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut function = Map::new();
            function.insert("name".into(), json!(tool.name));
            function.insert("description".into(), json!(tool.description));
            function.insert("parameters".into(), tool.parameters.clone());
            // Only include strict if provider supports it. Some reject unknown fields.
            if compat.supports_strict_mode {
                function.insert("strict".into(), json!(false));
            }
            json!({
                "type": "function",
                "function": Value::Object(function),
            })
        })
        .collect()
}

pub(crate) fn parse_chunk_usage(
    raw_usage: &Value,
    model: &Model,
    cache_write_cost: Option<f64>,
) -> Usage {
    let get_u64 = |value: &Value| value.as_u64().unwrap_or(0);
    let prompt_tokens = raw_usage.get("prompt_tokens").map_or(0, get_u64);
    let reported_cached_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .map(get_u64)
        .or_else(|| raw_usage.get("prompt_cache_hit_tokens").map(get_u64))
        .unwrap_or(0);
    let cache_write_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cache_write_tokens"))
        .map_or(0, get_u64);

    // Normalize to the provider-layer usage accounting semantics:
    // - cacheRead: hits from cache created by previous requests only
    // - cacheWrite: tokens written to cache in this request
    // Some OpenAI-compatible providers (observed on OpenRouter) report cached_tokens as (previous
    // hits + current writes). Remove cacheWrite from cacheRead in that case.
    let cache_read_tokens = if cache_write_tokens > 0 {
        reported_cached_tokens.saturating_sub(cache_write_tokens)
    } else {
        reported_cached_tokens
    };

    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    // OpenAI completion_tokens already includes reasoning_tokens.
    let output_tokens = raw_usage.get("completion_tokens").map_or(0, get_u64);
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
        cost: UsageCost::default(),
    };
    calculate_cost(
        model,
        &mut usage,
        cache_write_cost
            .map(|cache_write| CostOverrides {
                cache_write: Some(cache_write),
            })
            .as_ref(),
    );
    // OpenRouter reports billing truth in usage, already priced by the endpoint and service tier
    // that served the request (https://openrouter.ai/docs/api-reference/overview). Trust it over
    // the catalog-rate estimate, scaling the component breakdown to match.
    let reported_cost = if model.provider == "openrouter" {
        openrouter_reported_cost(raw_usage)
    } else {
        None
    };
    // Token counts sit far below f64's 2^53 exact-integer range; the cost apportioning math is f64 by design.
    #[allow(clippy::cast_precision_loss)]
    if let Some(reported_cost) = reported_cost {
        if usage.cost.total.as_f64() > 0.0 {
            let scale = reported_cost / usage.cost.total.as_f64();
            usage.cost.input = (usage.cost.input.as_f64() * scale).into();
            usage.cost.output = (usage.cost.output.as_f64() * scale).into();
            usage.cost.cache_read = (usage.cost.cache_read.as_f64() * scale).into();
            usage.cost.cache_write = (usage.cost.cache_write.as_f64() * scale).into();
        } else if usage.total_tokens > 0 {
            // No catalog rates to apportion by: attribute by token counts instead.
            let total_tokens = usage.total_tokens as f64;
            usage.cost.input = (reported_cost * usage.input as f64 / total_tokens).into();
            usage.cost.output = (reported_cost * usage.output as f64 / total_tokens).into();
            usage.cost.cache_read = (reported_cost * usage.cache_read as f64 / total_tokens).into();
            usage.cost.cache_write =
                (reported_cost * usage.cache_write as f64 / total_tokens).into();
        }
        usage.cost.total = reported_cost.into();
    }
    usage
}

/// The user's real spend for an `OpenRouter` request, or `None` to keep the catalog estimate.
/// `usage.cost` only carries what `OpenRouter` charged the account's credits: for BYOK requests
/// that is just `OpenRouter`'s fee, so real spend is the upstream provider's bill plus that fee. A
/// cost of 0 can mean not-billed-via-credits (e.g. `:free` endpoints) rather than free, so it keeps
/// the catalog estimate.
fn openrouter_reported_cost(raw_usage: &Value) -> Option<f64> {
    let credits = raw_usage
        .get("cost")
        .and_then(Value::as_f64)
        .filter(|cost| *cost > 0.0);
    if raw_usage.get("is_byok").and_then(Value::as_bool) == Some(true) {
        let upstream = raw_usage
            .get("cost_details")
            .and_then(|details| details.get("upstream_inference_cost"))
            .and_then(Value::as_f64);
        return upstream
            .filter(|upstream| *upstream > 0.0)
            .map(|upstream| upstream + credits.unwrap_or(0.0));
    }
    credits
}

pub(crate) fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_string()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_string()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, ModelInput};
    use pa_types::JsNumber;

    fn model(provider: &str, input: f64, output: f64) -> Model {
        Model {
            id: "m".into(),
            name: "m".into(),
            api: "openai-completions".into(),
            provider: provider.into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![ModelInput::Text],
            cost: ModelCost {
                input: JsNumber::from(input),
                output: JsNumber::from(output),
                cache_read: JsNumber::from(0.0),
                cache_write: JsNumber::from(0.0),
            },
            context_window: 128_000,
            max_tokens: 8192,
            max_tokens_explicit: false,
            featured: None,
            headers: None,
            compat: None,
        }
    }

    enum ByokBilling {
        NotByok,
        FeeOnly,
        WithUpstreamBill(f64),
    }

    // Test-only helper; adapting its signature and call sites would churn test fixtures.
    #[allow(clippy::needless_pass_by_value)]
    fn raw_usage(cost: f64, byok: ByokBilling) -> Value {
        let mut raw = json!({
            "prompt_tokens": 50_000,
            "completion_tokens": 50_000,
            "total_tokens": 100_000,
            "cost": cost,
        });
        match byok {
            ByokBilling::NotByok => {}
            ByokBilling::FeeOnly => raw["is_byok"] = json!(true),
            ByokBilling::WithUpstreamBill(upstream) => {
                raw["is_byok"] = json!(true);
                raw["cost_details"] = json!({ "upstream_inference_cost": upstream });
            }
        }
        raw
    }

    // --- issue #9: prior-turn reasoning strip ---

    fn strip_model() -> Model {
        let mut model = model("midagent", 0.5, 0.5);
        model.reasoning = true;
        model
    }

    fn compat_with_strip(strip: bool) -> ResolvedCompat {
        let mut compat = crate::providers::openai_completions::detect_compat(&strip_model());
        compat.strip_prior_turn_reasoning = strip;
        compat
    }

    fn ctx_user(text: &str) -> Message {
        Message::User(crate::types::UserMessage {
            content: UserMessageContent::Text(text.to_string()),
            timestamp: 0,
            rest: Map::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn assistant_msg(
        thinking: Option<(&str, Option<&str>)>,
        text: Option<&str>,
        tool_call: Option<(&str, Option<&str>)>,
        model_id: &str,
    ) -> Message {
        let mut content: Vec<AssistantContent> = Vec::new();
        if let Some((thinking, signature)) = thinking {
            content.push(AssistantContent::Thinking(ThinkingContent {
                thinking: thinking.to_string(),
                thinking_signature: signature.map(str::to_string),
                redacted: None,
                rest: Map::default(),
            }));
        }
        if let Some(text) = text {
            content.push(AssistantContent::Text(TextContent {
                text: text.to_string(),
                text_signature: None,
                rest: Map::default(),
            }));
        }
        if let Some((id, thought_signature)) = tool_call {
            content.push(AssistantContent::ToolCall(ToolCall {
                id: id.to_string(),
                name: "echo".to_string(),
                arguments: Map::default(),
                thought_signature: thought_signature.map(str::to_string),
                rest: Map::default(),
            }));
        }
        Message::Assistant(crate::types::AssistantMessage {
            content,
            api: "openai-completions".into(),
            provider: "midagent".into(),
            model: model_id.to_string(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn ctx_tool_result(id: &str) -> Message {
        Message::ToolResult(crate::types::ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: "echo".to_string(),
            content: vec![UserOrToolContent::Text(TextContent {
                text: "result".to_string(),
                text_signature: None,
                rest: Map::default(),
            })],
            details: None,
            is_error: false,
            timestamp: 0,
            rest: Map::default(),
        })
    }

    fn two_turn_context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![
                ctx_user("turn one"),
                assistant_msg(
                    Some(("old reasoning", Some("reasoning"))),
                    Some("old answer"),
                    Some(("call-1", Some(r#"[{"type":"reasoning"}]"#))),
                    "m",
                ),
                ctx_tool_result("call-1"),
                ctx_user("turn two"),
                assistant_msg(
                    Some(("current reasoning", Some("reasoning"))),
                    None,
                    Some(("call-2", Some(r#"[{"type":"reasoning"}]"#))),
                    "m",
                ),
                ctx_tool_result("call-2"),
            ],
            tools: None,
        }
    }

    fn message_json(params: &[Value], index: usize) -> &Value {
        params
            .get(index)
            .unwrap_or_else(|| panic!("no message at index {index}"))
    }

    #[test]
    fn strip_removes_old_turn_reasoning_and_keeps_current_chain() {
        let model = strip_model();
        let context = two_turn_context();

        // Flag on: the first (old-turn) assistant loses its reasoning replay;
        // the second (current tool-call chain) keeps it.
        let params = convert_messages(&model, &context, &compat_with_strip(true));
        let old = message_json(&params, 1);
        assert_eq!(old["role"], "assistant");
        assert!(old.get("reasoning").is_none(), "old reasoning stripped");
        assert!(
            old.get("reasoning_details").is_none(),
            "old tool-call reasoning_details stripped"
        );
        assert_eq!(old["content"], "old answer");
        let current = message_json(&params, 4);
        assert_eq!(current["role"], "assistant");
        assert_eq!(current["reasoning"], "current reasoning");
        assert!(current.get("reasoning_details").is_some());

        // Flag off (default): full replay, unchanged behavior.
        let params = convert_messages(&model, &context, &compat_with_strip(false));
        let old = message_json(&params, 1);
        assert_eq!(old["reasoning"], "old reasoning");
        assert!(old.get("reasoning_details").is_some());
    }

    #[test]
    fn strip_removes_cross_model_thinking_before_the_transform_folds_it_into_text() {
        // B1 regression: cross-model unsigned thinking becomes visible text in
        // the transform, so the strip must run before it.
        let model = strip_model();
        let context = Context {
            system_prompt: None,
            messages: vec![
                ctx_user("turn one"),
                assistant_msg(
                    Some(("cross model reasoning", None)),
                    Some("old answer"),
                    None,
                    "other-model",
                ),
                ctx_user("turn two"),
            ],
            tools: None,
        };

        let params = convert_messages(&model, &context, &compat_with_strip(true));
        let wire = serde_json::to_string(&params).expect("serializable");
        assert!(
            !wire.contains("cross model reasoning"),
            "cross-model thinking must not reach the wire in any form"
        );

        let params = convert_messages(&model, &context, &compat_with_strip(false));
        let wire = serde_json::to_string(&params).expect("serializable");
        assert!(
            wire.contains("cross model reasoning"),
            "flag off keeps the cross-model thinking (as text)"
        );
    }

    #[test]
    fn strip_drops_a_thinking_only_assistant_message() {
        let model = strip_model();
        let context = Context {
            system_prompt: None,
            messages: vec![
                ctx_user("turn one"),
                assistant_msg(Some(("only thinking", None)), None, None, "other-model"),
                ctx_user("turn two"),
            ],
            tools: None,
        };

        let params = convert_messages(&model, &context, &compat_with_strip(true));
        // The stripped assistant is empty and falls into the skip guard.
        assert_eq!(params.len(), 2);
        assert_eq!(params[0]["role"], "user");
        assert_eq!(params[1]["role"], "user");
    }

    #[test]
    fn get_compat_resolves_strip_prior_turn_reasoning_from_models_json() {
        use crate::types::{CompatKind, ModelCompat, OpenAiCompletionsCompat};
        let mut model = strip_model();
        model.compat = Some(ModelCompat::from_kind(CompatKind::OpenAiCompletions(
            Box::new(OpenAiCompletionsCompat {
                strip_prior_turn_reasoning: Some(true),
                ..Default::default()
            }),
        )));
        assert!(
            crate::providers::openai_completions::get_compat(&model).strip_prior_turn_reasoning
        );
        // Default off without the models.json override.
        assert!(
            !crate::providers::openai_completions::get_compat(&strip_model())
                .strip_prior_turn_reasoning
        );
    }

    #[test]
    fn openrouter_zero_reported_cost_keeps_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // A cost of 0 can mean not-billed-via-credits (:free endpoints) rather than free, so the
        // catalog estimate stays.
        let usage = parse_chunk_usage(&raw_usage(0.0, ByokBilling::NotByok), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn openrouter_byok_without_upstream_keeps_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // BYOK credits are only OpenRouter's fee; without the upstream bill the real spend is
        // unknown, so the catalog estimate stays.
        let usage = parse_chunk_usage(&raw_usage(0.003, ByokBilling::FeeOnly), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn openrouter_byok_upstream_cost_replaces_catalog_estimate() {
        let model = model("openrouter", 0.5, 0.5);
        // Credits charged by OpenRouter plus the upstream provider's bill.
        let usage = parse_chunk_usage(
            &raw_usage(0.003, ByokBilling::WithUpstreamBill(0.2)),
            &model,
            None,
        );
        assert!((usage.cost.input.as_f64() - 0.1015).abs() < 1e-9);
        assert!((usage.cost.output.as_f64() - 0.1015).abs() < 1e-9);
        assert!((usage.cost.total.as_f64() - 0.203).abs() < 1e-9);
    }

    #[test]
    fn openrouter_reported_cost_apportioned_by_tokens_without_rates() {
        let model = model("openrouter", 0.0, 0.0);
        let raw = json!({"prompt_tokens": 40_000, "completion_tokens": 10_000, "total_tokens": 50_000, "cost": 0.15});
        let usage = parse_chunk_usage(&raw, &model, None);
        assert!((usage.cost.input.as_f64() - 0.12).abs() < 1e-9);
        assert!((usage.cost.output.as_f64() - 0.03).abs() < 1e-9);
        assert!((usage.cost.cache_read.as_f64() - 0.0).abs() < 1e-9);
        assert!((usage.cost.total.as_f64() - 0.15).abs() < 1e-9);
    }

    #[test]
    fn non_openrouter_provider_ignores_reported_cost() {
        let model = model("zai", 0.5, 0.5);
        let usage = parse_chunk_usage(&raw_usage(0.07, ByokBilling::NotByok), &model, None);
        assert!((usage.cost.total.as_f64() - 0.05).abs() < 1e-9);
    }
}
