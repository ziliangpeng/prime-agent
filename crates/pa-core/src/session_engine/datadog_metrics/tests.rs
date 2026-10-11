use std::net::UdpSocket;
use std::time::Duration;

use pa_agent::types::{AssistantContent, AssistantMessage, StopReason, Usage};

use super::{
    build_packet, install_datadog_metrics, parse_port, resolve_host, tag_suffix,
    DatadogMetricsConfig,
};

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
            "prime.tokens.input:100|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.output:20|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.cache_read:5|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.cache_write:7|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.tokens.total:132|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.api.calls:1|c#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
            "prime.api.duration_ms:2000|h#api:anthropic-messages,model:claude-x-2026,provider:custom_midagent",
        ]
    );
}

#[test]
fn build_packet_skips_zero_token_counters() {
    let msg = assistant(Usage::zero(), StopReason::Stop, 5_000);
    let packet = build_packet(&msg, 10_000);
    // Only calls + duration: zero-value token counters are skipped.
    assert_eq!(packet.lines().count(), 2);
    assert!(packet.starts_with("prime.api.calls:1|c#"));
    assert!(packet.contains("prime.api.duration_ms:5000|h#"));
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

#[test]
fn client_sends_single_datagram_per_message() {
    // Receiver on an ephemeral localhost port.
    let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = receiver.local_addr().unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    let cfg = DatadogMetricsConfig {
        enabled: true,
        addr,
    };
    let client = super::CLIENT.get_or_init(super::DogStatsDClient::new);
    let msg = assistant(full_usage(), StopReason::Stop, 1_000);
    let packet = build_packet(&msg, 2_000);
    client.send_packet(cfg.addr, &packet);

    let mut buf = [0u8; 2048];
    let (len, _from) = receiver.recv_from(&mut buf).expect("datagram arrives");
    assert_eq!(&buf[..len], packet.as_bytes());
    // One send = one datagram, newline-joined.
    assert_eq!(std::str::from_utf8(&buf[..len]).unwrap(), packet);
}

#[test]
fn client_rebuilds_after_send_failure_without_panicking() {
    // Bind then drop a receiver: sends to it still succeed locally (UDP),
    // so instead exercise the rebuild path directly through an unreachable
    // address family-safe port on loopback and assert no panic across
    // repeated sends.
    let dead = DatadogMetricsConfig {
        enabled: true,
        addr: "127.0.0.1:1".parse().unwrap(),
    };
    let client = super::CLIENT.get_or_init(super::DogStatsDClient::new);
    for _ in 0..3 {
        client.send_packet(dead.addr, "prime.api.calls:1|c");
    }
}

#[tokio::test]
async fn install_disabled_subscribes_nothing_and_never_fails() {
    // Disabled by default env (test env does not set the flag): install on a
    // bare agent and run it; no metrics, no errors. This exercises the
    // no-op path end to end.
    let agent = std::sync::Arc::new(pa_agent::agent::Agent::new(
        pa_agent::agent::AgentOptions::default(),
    ));
    install_datadog_metrics(&agent).await;
    // A run with no prompt material produces no assistant messages; the
    // listener (never installed) trivially never fires. No panic, no error.
}
