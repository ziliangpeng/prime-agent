//! Datadog token metrics via a `DogStatsD` agent subscription (fork feature).
//!
//! Port of the TS-era built-in extension `datadog-tokens.ts`: per-API-call
//! token counters and a request-duration histogram, emitted as one newline-
//! joined `DogStatsD` datagram per assistant message over UDP. Labeled with the
//! raw `model`/`provider`/`api` strings, so existing Datadog dashboards keep
//! their TS-era granularity. Opt-in via `PRIME_AGENT_DATADOG_METRICS=1`;
//! disabled (the default) the install is one env read and returns.
//!
//! This is a fork-only observability path: it subscribes to
//! [`AgentEvent::MessageEnd`] independently and never touches the
//! telemetry event stream (whose sanitize catalog would strip the raw
//! model/provider/api labels).

use std::net::{IpAddr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex, OnceLock};

use pa_agent::types::{AgentEvent, AgentMessage, Message, StopReason};

/// Enable flag: only the exact value `"1"` turns metrics on.
const ENV_ENABLE: &str = "PRIME_AGENT_DATADOG_METRICS";
/// `DogStatsD` agent host (IP or resolvable hostname). Default `127.0.0.1`.
const ENV_HOST: &str = "PRIME_AGENT_DATADOG_AGENT_HOST";
/// `DogStatsD` agent port, 1-65535; malformed values fall back to the default.
const ENV_PORT: &str = "PRIME_AGENT_DATADOG_AGENT_PORT";

const DEFAULT_HOST: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
const DEFAULT_PORT: u16 = 8125;

/// Duration sanity clamp: skip durations above 1h (exotic provider
/// timestamps). Mirrors the TS `MAX_DURATION_MS`.
const MAX_DURATION_MS: i64 = 60 * 60 * 1000;

/// Characters that break the `DogStatsD` wire format (pipe, comma, colon,
/// whitespace).
const TAG_SANITIZE_CHARS: &[char] = &['|', ':', ',', ' ', '\t', '\r', '\n'];

/// Datadog metrics configuration, read once per install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatadogMetricsConfig {
    /// `PRIME_AGENT_DATADOG_METRICS == "1"`.
    pub enabled: bool,
    /// Resolved `DogStatsD` socket address. Hostname resolution happens here,
    /// once, so the send path never blocks on DNS inside an async listener.
    pub addr: SocketAddr,
}

impl DatadogMetricsConfig {
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_env_with(|name: &str| std::env::var(name))
    }

    fn from_env_with<F>(get: F) -> Self
    where
        F: Fn(&str) -> Result<String, std::env::VarError>,
    {
        let enabled = matches!(get(ENV_ENABLE).as_deref(), Ok("1"));
        let host_raw = get(ENV_HOST)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_HOST.to_string());
        let port = parse_port(get(ENV_PORT).ok().as_deref());
        let addr = resolve_host(&host_raw, port);
        Self { enabled, addr }
    }
}

/// Parse a port env value (1-65535), falling back when absent or malformed.
fn parse_port(raw: Option<&str>) -> u16 {
    let Some(raw) = raw else {
        return DEFAULT_PORT;
    };
    raw.trim()
        .parse::<u16>()
        .ok()
        .filter(|p| *p > 0)
        .unwrap_or(DEFAULT_PORT)
}

/// Resolve the configured host to a socket address once, at config time.
/// Unparseable hosts and failed DNS fall back to the default loopback
/// address: metrics are best-effort and must never fail session creation.
fn resolve_host(host: &str, port: u16) -> SocketAddr {
    if let Ok(ip) = host.parse::<IpAddr>() {
        return SocketAddr::new(ip, port);
    }
    host.to_socket_addrs()
        .ok()
        .and_then(|mut addrs| addrs.next())
        .unwrap_or_else(|| SocketAddr::new(DEFAULT_HOST, port))
}

/// Process-wide `DogStatsD` client: ONE UDP socket per process no matter how
/// many sessions install the subscription (a daemon with session churn must
/// not accumulate sockets — an improvement over the TS per-session client).
/// Send failures drop the socket; the next send rebuilds it. Fail-open: all
/// errors are swallowed — observability must never break the agent.
struct DogStatsDClient {
    socket: Mutex<Option<UdpSocket>>,
}

/// Local any-address matching the target's address family: an IPv6-bound
/// socket cannot send to an IPv4 agent (and vice versa), so a hardcoded
/// IPv4 bind made every IPv6-resolved agent target deterministically
/// unreachable. Bind the family of the resolved destination.
fn local_bind_addr(target: SocketAddr) -> SocketAddr {
    let ip = match target {
        SocketAddr::V4(_) => IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
        SocketAddr::V6(_) => IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED),
    };
    SocketAddr::new(ip, 0)
}

impl DogStatsDClient {
    const fn new() -> Self {
        DogStatsDClient {
            socket: Mutex::new(None),
        }
    }

    /// Send one datagram, best-effort. Never panics: a send error (including
    /// the `Err` surfaced by a later `send_to` after ICMP port-unreachable)
    /// drops the socket for rebuild on the next call.
    fn send_packet(&self, addr: SocketAddr, packet: &str) {
        let mut guard = self
            .socket
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_none() {
            match UdpSocket::bind(local_bind_addr(addr)) {
                Ok(sock) => *guard = Some(sock),
                Err(_) => return,
            }
        }
        let Some(sock) = guard.as_ref() else {
            return;
        };
        if sock.send_to(packet.as_bytes(), addr).is_err() {
            // Drop the broken socket; the next metric creates a fresh one.
            *guard = None;
        }
    }
}

/// Process-level singleton client (shared across all sessions in this
/// process).
static CLIENT: OnceLock<DogStatsDClient> = OnceLock::new();

fn client() -> &'static DogStatsDClient {
    CLIENT.get_or_init(DogStatsDClient::new)
}

/// Sanitize one tag value for the `DogStatsD` wire format: replace the
/// format-breaking characters with `_`, and default empty values to
/// `unknown`.
fn sanitize_tag_value(value: &str) -> String {
    if value.is_empty() {
        "unknown".to_string()
    } else {
        value.replace(TAG_SANITIZE_CHARS, "_")
    }
}

/// Build the sorted `api,model,provider` tag suffix (`#tags`-joined).
fn tag_suffix(model: &str, response_model: Option<&str>, provider: &str, api: &str) -> String {
    // Keys in sorted order: api, model, provider.
    let api = sanitize_tag_value(api);
    let model = sanitize_tag_value(response_model.filter(|m| !m.is_empty()).unwrap_or(model));
    let provider = sanitize_tag_value(provider);
    format!("#api:{api},model:{model},provider:{provider}")
}

/// Build the newline-joined `DogStatsD` datagram for one assistant message.
/// Always contains the `prime.api.calls` line; token counters are skipped
/// when zero or when the run aborted; the duration histogram is clamped to
/// `0 < duration <= 1h`.
#[must_use]
pub fn build_packet(msg: &pa_agent::types::AssistantMessage, now_ms: i64) -> String {
    let suffix = tag_suffix(
        &msg.model,
        msg.response_model.as_deref(),
        &msg.provider,
        &msg.api,
    );

    let mut lines: Vec<String> = Vec::new();
    // Aborted messages carry partial usage; skip token counters but still
    // count the attempt.
    if msg.stop_reason != StopReason::Aborted {
        let counts: [(&str, u64); 5] = [
            ("prime.tokens.input", msg.usage.input),
            ("prime.tokens.output", msg.usage.output),
            ("prime.tokens.cache_read", msg.usage.cache_read),
            ("prime.tokens.cache_write", msg.usage.cache_write),
            ("prime.tokens.total", msg.usage.total_tokens),
        ];
        for (metric, value) in counts {
            if value > 0 {
                lines.push(format!("{metric}:{value}|c|{suffix}"));
            }
        }
    }

    // Request counter: 1 per assistant message (i.e. per API call).
    lines.push(format!("prime.api.calls:1|c|{suffix}"));

    // Duration histogram (ms), from request start to message end.
    let duration = now_ms - msg.timestamp;
    if duration > 0 && duration <= MAX_DURATION_MS {
        lines.push(format!("prime.api.duration_ms:{duration}|h|{suffix}"));
    }

    lines.join("\n")
}

/// Install the Datadog token-metrics subscription on the session's agent.
///
/// No-op (one env read, zero sockets, zero subscriptions) unless
/// `PRIME_AGENT_DATADOG_METRICS=1`. Installed outside the telemetry
/// depth-0 gate: RLM child sessions run in the same process and report like
/// any other session (matching the TS extension, which every process load
/// installed).
///
/// The listener never returns `Err` and never panics — a failing listener
/// fails the run, and metrics must be fail-open. The `Subscription` handle
/// is dropped on purpose: dropping it does not unsubscribe, and the
/// listener's lifetime is the agent's.
pub async fn install_datadog_metrics(agent: &Arc<pa_agent::agent::Agent>) {
    install_with_config(agent, DatadogMetricsConfig::from_env()).await;
}

/// Config-injected install so tests can drive the enabled path against a
/// local UDP receiver without mutating process env vars.
async fn install_with_config(agent: &Arc<pa_agent::agent::Agent>, config: DatadogMetricsConfig) {
    if !config.enabled {
        return;
    }
    let client = client();
    agent
        .subscribe(move |event, _signal| {
            let config = config.clone();
            Box::pin(async move {
                if let AgentEvent::MessageEnd {
                    message: AgentMessage::Standard(Message::Assistant(msg)),
                } = event
                {
                    let packet = build_packet(&msg, pa_agent::now_ms());
                    client.send_packet(config.addr, &packet);
                }
                Ok(())
            })
        })
        .await;
}

#[cfg(test)]
mod tests;
