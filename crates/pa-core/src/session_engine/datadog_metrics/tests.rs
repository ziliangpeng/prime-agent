use std::net::UdpSocket;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use pa_agent::agent::{Agent, AgentInitialState, AgentOptions};
use pa_agent::scripted::{ScriptStep, ScriptedProvider, ScriptedTurn};
use pa_agent::stream::AssistantMessageEvent;
use pa_agent::types::{AssistantContent, AssistantMessage, Model, StopReason, Usage};

use super::{
    build_packet, install_datadog_metrics, install_with_config, parse_port, resolve_host,
    tag_suffix, DatadogMetricsConfig,
};

/// Serializes tests that send through the process-wide [`super::CLIENT`]
/// singleton: two tests sharing one UDP socket in parallel can steal each
/// other's datagrams (and on Linux, ICMP port-unreachable feedback on the
/// shared socket surfaces as a send error in the *other* test). Every test
/// that touches the singleton (or binds a metrics receiver and drives the
/// real install path) holds this lock.
fn udp_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn assistant(usage: Usage, stop_reason: StopReason, timestamp: i64) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::Text(pa_agent::types::TextContent {
            text: "ok".to_string(),
            text_signature: None,
        })],
        api: "anthropic-messages".to_string(),
        provider: "custom:midagent".to_string(),
        model: "claude-x".to_string(),
        response_model: Some("claude-x-2026".to_string()),
        response_id: None,
        diagnostics: None,
        usage,
        stop_reason,
        stop_reason_raw: None,
        error_message: None,
        timestamp,
    }
}

fn env_map<'a>(
    vars: &'a [(&'a str, &'a str)],
) -> impl Fn(&str) -> Result<String, std::env::VarError> + 'a {
    move |name: &str| {
        vars.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.to_string())
            .ok_or(std::env::VarError::NotPresent)
    }
}

fn full_usage() -> Usage {
    Usage {
        input: 100,
        output: 20,
        cache_read: 5,
        cache_write: 7,
        total_tokens: 132,
        cost: pa_agent::types::UsageCost::default(),
    }
}

#[test]
fn config_defaults_to_disabled() {
    let cfg = DatadogMetricsConfig::from_env_with(env_map(&[]));
    assert!(!cfg.enabled);
    assert_eq!(cfg.addr.port(), 8125);
    assert_eq!(cfg.addr.ip().to_string(), "127.0.0.1");
}

#[test]
fn config_enables_only_on_exact_one() {
    assert!(
        DatadogMetricsConfig::from_env_with(env_map(&[("PRIME_AGENT_DATADOG_METRICS", "1",)]))
            .enabled
    );
    for raw in ["true", "yes", ""] {
        assert!(
            !DatadogMetricsConfig::from_env_with(env_map(&[("PRIME_AGENT_DATADOG_METRICS", raw,)]))
                .enabled
        );
    }
}

#[test]
fn config_port_parsing() {
    let cfg = DatadogMetricsConfig::from_env_with(env_map(&[
        ("PRIME_AGENT_DATADOG_METRICS", "1"),
        ("PRIME_AGENT_DATADOG_AGENT_PORT", "9999"),
    ]));
    assert_eq!(cfg.addr.port(), 9999);

    for bad in ["not-a-port", "0", "99999", "-1"] {
        let cfg = DatadogMetricsConfig::from_env_with(env_map(&[(
            "PRIME_AGENT_DATADOG_AGENT_PORT",
            bad,
        )]));
        assert_eq!(cfg.addr.port(), 8125, "port {bad:?} must fall back");
    }
}

#[test]
fn config_host_resolution_falls_back_to_loopback() {
    let cfg = DatadogMetricsConfig::from_env_with(env_map(&[(
        "PRIME_AGENT_DATADOG_AGENT_HOST",
        "not a host name!",
    )]));
    assert_eq!(cfg.addr.ip().to_string(), "127.0.0.1");
}

#[test]
fn parse_port_direct() {
    assert_eq!(parse_port(None), 8125);
    assert_eq!(parse_port(Some("1")), 1);
    assert_eq!(parse_port(Some("65535")), 65535);
    assert_eq!(parse_port(Some("0")), 8125);
    assert_eq!(parse_port(Some("x")), 8125);
}

#[test]
fn resolve_host_accepts_ipv6() {
    let addr = resolve_host("::1", 9999);
    assert_eq!(addr.port(), 9999);
    assert!(addr.is_ipv6());
}

#[test]
fn tag_suffix_sanitizes_and_prefers_response_model() {
    let suffix = tag_suffix(
        "claude x",
        Some("claude|real"),
        "custom:midagent",
        "anthropic-messages",
    );
    assert_eq!(
        suffix,
        "#api:anthropic-messages,model:claude_real,provider:custom_midagent"
    );
}

#[test]
fn tag_suffix_unknown_defaults() {
    let suffix = tag_suffix("", None, "", "");
    assert_eq!(suffix, "#api:unknown,model:unknown,provider:unknown");
}

#[test]
fn build_packet_full_message() {
    let now = 10_000;
    let msg = assistant(full_usage(), StopReason::Stop, 8_000);
    let packet = build_packet(&msg, now);
    let lines: Vec<&str> = packet.split('\n').collect();
    assert_eq!(
        lines,
        [
            "prime.tokens.input:100|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.output:20|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.cache_read:5|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.cache_write:7|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.total:132|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.api.calls:1|c|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.api.duration_ms:2000|h|#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
        ]
    );
}

#[test]
fn build_packet_skips_zero_token_counters() {
    let msg = assistant(Usage::zero(), StopReason::Stop, 5_000);
    let packet = build_packet(&msg, 10_000);
    // Only calls + duration: zero-value token counters are skipped.
    assert_eq!(packet.lines().count(), 2);
    assert!(packet.starts_with("prime.api.calls:1|c|#"));
    assert!(packet.contains("prime.api.duration_ms:5000|h|#"));
}

#[test]
fn build_packet_aborted_keeps_calls_and_duration() {
    let msg = assistant(full_usage(), StopReason::Aborted, 9_000);
    let packet = build_packet(&msg, 10_000);
    assert_eq!(packet.lines().count(), 2);
    assert!(!packet.contains("prime.tokens."));
    assert!(packet.contains("prime.api.calls:1|c"));
    assert!(packet.contains("prime.api.duration_ms:1000|h"));
}

#[test]
fn build_packet_duration_clamped() {
    let msg = assistant(full_usage(), StopReason::Stop, 0);
    // duration > 1h -> skipped.
    let packet = build_packet(&msg, 60 * 60 * 1000 + 1);
    assert!(!packet.contains("duration_ms"));

    // duration == 0 (synthetic abort timestamp) -> skipped.
    let msg = assistant(full_usage(), StopReason::Stop, 10_000);
    let packet = build_packet(&msg, 10_000);
    assert!(!packet.contains("duration_ms"));

    // duration == exactly 1h -> kept.
    let msg = assistant(full_usage(), StopReason::Stop, 0);
    let packet = build_packet(&msg, 60 * 60 * 1000);
    assert!(packet.contains("prime.api.duration_ms:3600000|h"));
}

#[tokio::test]
async fn client_sends_single_datagram_per_message() {
    let _guard = udp_lock().lock().await;
    // Receiver on an ephemeral localhost port.
    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = receiver.local_addr().unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    let client = super::CLIENT.get_or_init(super::DogStatsDClient::new);
    let msg = assistant(full_usage(), StopReason::Stop, 1_000);
    let packet = build_packet(&msg, 2_000);
    client.send_packet(addr, &packet);

    let mut buf = [0u8; 2048];
    let (len, _from) = receiver.recv_from(&mut buf).expect("datagram arrives");
    assert_eq!(&buf[..len], packet.as_bytes());
    // One send = one datagram, newline-joined.
    assert_eq!(std::str::from_utf8(&buf[..len]).unwrap(), packet);
}

#[test]
fn client_rebuilds_after_send_failure_without_panicking() {
    // Controlled, traffic-free failure: a fresh LOCAL client (not the
    // process singleton) poisoned with an IPv6-bound socket cannot send to
    // an IPv4 address — the OS rejects the address-family mismatch
    // deterministically, no packets leave the machine (and no ICMP
    // port-unreachable can hit any shared socket). The failed send drops
    // the broken socket; the next send rebuilds one with the right family
    // and delivers to the live receiver.
    let client = super::DogStatsDClient::new();
    *client.socket.lock().unwrap() =
        Some(UdpSocket::bind("[::1]:0").expect("bind poisoned v6 socket"));

    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = receiver.local_addr().unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    let packet = "prime.api.calls:1|c|#api:test";
    // First send fails on the family mismatch and drops the broken socket.
    client.send_packet(addr, packet);
    assert!(
        client.socket.lock().unwrap().is_none(),
        "failed send must drop the broken socket"
    );
    // Second send rebuilds (IPv4 family) and delivers.
    client.send_packet(addr, packet);

    let mut buf = [0u8; 1024];
    let (len, _from) = receiver.recv_from(&mut buf).expect("rebuilt send delivers");
    assert_eq!(&buf[..len], packet.as_bytes());
    // Repeated sends after the rebuild keep working: no panic, still bound.
    client.send_packet(addr, packet);
    assert!(client.socket.lock().unwrap().is_some());
}

#[tokio::test]
async fn install_disabled_subscribes_nothing_and_never_fails() {
    let _guard = udp_lock().lock().await;
    // Disabled config: install is a no-op, a full scripted run produces an
    // assistant MessageEnd, and the receiver stays silent.
    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();

    let (agent, _provider) = scripted_agent(full_usage(), 2_000);
    install_datadog_metrics(&agent).await;
    agent.prompt("hi").await.expect("run completes");

    let mut buf = [0u8; 2048];
    assert!(
        receiver.recv_from(&mut buf).is_err(),
        "disabled install must not emit any datagram"
    );
}

#[tokio::test]
async fn install_enabled_end_to_end_sends_metrics_via_udp() {
    let _guard = udp_lock().lock().await;
    // The full enabled chain: install(enabled config) subscribes on the
    // agent; a scripted assistant message (full usage/model/provider) fires
    // MessageEnd; the listener builds the packet and the process-wide client
    // sends it over UDP loopback; the receiver sees every expected metric
    // line in the official DogStatsD wire format.
    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = receiver.local_addr().unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    let (agent, _provider) = scripted_agent(full_usage(), 1_500);
    install_with_config(
        &agent,
        DatadogMetricsConfig {
            enabled: true,
            addr,
        },
    )
    .await;
    agent.prompt("hi").await.expect("run completes");

    let mut buf = [0u8; 4096];
    let (len, _from) = receiver
        .recv_from(&mut buf)
        .expect("metrics datagram arrives");
    let packet = std::str::from_utf8(&buf[..len]).expect("datagram is UTF-8");
    let lines: Vec<&str> = packet.lines().collect();

    let tags = "#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent";
    assert_eq!(lines.len(), 7, "one line per emitted metric");
    assert_eq!(lines[0], format!("prime.tokens.input:100|c|{tags}"));
    assert_eq!(lines[1], format!("prime.tokens.output:20|c|{tags}"));
    assert_eq!(lines[2], format!("prime.tokens.cache_read:5|c|{tags}"));
    assert_eq!(lines[3], format!("prime.tokens.cache_write:7|c|{tags}"));
    assert_eq!(lines[4], format!("prime.tokens.total:132|c|{tags}"));
    assert_eq!(lines[5], format!("prime.api.calls:1|c|{tags}"));

    // Duration depends on wall-clock fire time; assert shape and plausibility.
    let expected_prefix = "prime.api.duration_ms:";
    assert!(
        lines[6].starts_with(expected_prefix) && lines[6].ends_with(&format!("|h|{tags}")),
        "duration line shape: {}",
        lines[6]
    );
    let value: i64 = lines[6]
        .trim_start_matches(expected_prefix)
        .split('|')
        .next()
        .unwrap()
        .parse()
        .expect("duration parses as integer");
    assert!(
        value > 0 && value <= 1_500_000,
        "duration {value} plausible"
    );

    // Every line conforms to the DogStatsD wire format:
    // `<metric.name>:<value>|<type>|#<tag:value,...>` (tags after a `|`,
    // per the official DogStatsD datagram spec).
    for line in &lines {
        assert_dogstatsd_wire_format(line);
    }
}

/// Assert one line matches the official `DogStatsD` datagram format
/// `metric.name:value|type|#tag:key:value,...` where `type` is one of
/// `c` (counter) or `h` (histogram) and each tag is `key:value`.
/// Reference: Datadog "`DogStatsD` datagram format" documentation.
fn assert_dogstatsd_wire_format(line: &str) {
    let (body, tags) = line.split_once('|').expect("value|type separator");
    let (name, value) = body.split_once(':').expect("metric:value");
    assert!(
        !name.is_empty() && name.contains('.'),
        "metric name {name:?}"
    );
    assert!(value.parse::<i64>().is_ok(), "value {value:?} is numeric");
    assert!(
        tags.starts_with("c|") || tags.starts_with("h|"),
        "type section {tags:?}"
    );
    let tag_section = tags.split_once('|').expect("type|#tags separator").1;
    assert!(tag_section.starts_with('#'), "tags open with #");
    let tags_body = &tag_section[1..];
    let tag_list: Vec<&str> = tags_body.split(',').collect();
    assert_ne!(tag_list.len(), 0);
    for tag in &tag_list {
        let (key, val) = tag.split_once(':').expect("tag key:value");
        assert!(!key.is_empty() && !val.is_empty(), "tag {tag:?}");
    }
    // Tags sorted (api, model, provider) keeps Datadog tag cardinality stable.
    let mut sorted = tag_list.clone();
    sorted.sort_unstable();
    assert_eq!(tag_list, sorted, "tags emitted in sorted order");
}

#[test]
fn build_packet_matches_dogstatsd_wire_format() {
    let msg = assistant(full_usage(), StopReason::Stop, 5_000);
    let packet = build_packet(&msg, 10_000);
    for line in packet.lines() {
        assert_dogstatsd_wire_format(line);
    }
}

/// A bare agent wired to a scripted provider that answers one assistant
/// message carrying the given usage and `now - age_ms` timestamp.
fn scripted_agent(usage: Usage, age_ms: i64) -> (Arc<Agent>, Arc<ScriptedProvider>) {
    let model = Model {
        id: "claude-x".to_string(),
        name: "Claude X".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "custom:midagent".to_string(),
        base_url: String::new(),
        reasoning: false,
        cost: pa_agent::types::UsageCost::default(),
        context_window: 0,
        max_tokens: 0,
        max_tokens_explicit: false,
    };
    let provider = Arc::new(ScriptedProvider::new(model.clone()));
    let base = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::zero(),
        stop_reason: StopReason::Stop,
        stop_reason_raw: None,
        error_message: None,
        timestamp: pa_agent::now_ms() - age_ms,
    };
    let final_message = AssistantMessage {
        content: vec![AssistantContent::Text(pa_agent::types::TextContent {
            text: "ok".to_string(),
            text_signature: None,
        })],
        response_model: Some("claude-x-2026".to_string()),
        usage,
        ..base.clone()
    };
    provider.push_turn(ScriptedTurn::Events(vec![
        ScriptStep::Event(Box::new(AssistantMessageEvent::Start { partial: base })),
        ScriptStep::Event(Box::new(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: final_message,
        })),
    ]));
    let agent = Arc::new(Agent::new(AgentOptions {
        initial_state: AgentInitialState {
            model: Some(model),
            ..Default::default()
        },
        stream_fn: Some(provider.stream_fn()),
        ..Default::default()
    }));
    (agent, provider)
}
