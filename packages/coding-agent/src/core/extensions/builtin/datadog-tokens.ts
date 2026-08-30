/**
 * Built-in Datadog token-metrics extension.
 *
 * Emits per-API-call token counts as DogStatsD metrics labeled by model,
 * provider, and API type — the Prime Agent equivalent of Hermes'
 * `plugins/observability/datadog_tokens` plugin. Uses non-blocking UDP to
 * a local DogStatsD agent (default 127.0.0.1:8125).
 *
 * Metrics emitted (all counters, Datadog auto-rollups by time window):
 *
 *   prime.tokens.input        — prompt/input tokens per API call
 *   prime.tokens.output       — completion/output tokens per API call
 *   prime.tokens.cache_read   — cache-read tokens (cached prefix replay)
 *   prime.tokens.cache_write  — cache-write tokens (newly cached prefix)
 *   prime.tokens.total        — total tokens per API call
 *   prime.api.calls           — 1 per assistant message (request counter)
 *   prime.api.duration_ms     — request-to-message-end duration (HISTOGRAM)
 *
 * All metrics tagged with:
 *   model     — the model that served the request (responseModel preferred)
 *   provider  — the provider string (openai, anthropic, custom:midagent, ...)
 *   api       — the API protocol (openai-completions, anthropic-messages, ...)
 *
 * Disabled by default. Enable with PRIME_AGENT_DATADOG_METRICS=1.
 * Optional env vars:
 *   PRIME_AGENT_DATADOG_AGENT_HOST  — DogStatsD host (default: 127.0.0.1)
 *   PRIME_AGENT_DATADOG_AGENT_PORT  — DogStatsD port (default: 8125)
 *
 * Fail-open: UDP sends never block the agent loop and never throw into it.
 *
 * Lifetime: the factory runs per session load and reads env at that moment.
 * Each session gets its own client; the UDP socket is created lazily on the
 * first metric, unref'd so it never keeps the process alive, and closed on
 * `session_shutdown`.
 */

import { createSocket, type Socket } from "node:dgram";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionFactory } from "../types.js";

/** Characters that break the DogStatsD wire format (pipe, comma, colon, whitespace). */
const TAG_SANITIZE_RE = /[|:,\s\r\n]/g;

/** Duration sanity clamp: skip durations above 1h (exotic provider timestamps). */
const MAX_DURATION_MS = 60 * 60 * 1000;

interface DogStatsDClient {
	send(packet: string): void;
	close(): void;
}

/** Parse a port env value (1-65535), falling back when absent or malformed. */
export function parsePortEnv(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) && parsed > 0 && parsed <= 65535 ? parsed : fallback;
}

/**
 * Create a lazily-initialized DogStatsD client over a UDP socket.
 * Returns null when metrics are disabled via env. The socket is unref'd so
 * it never keeps the process alive, and send failures are swallowed —
 * observability must never break the agent.
 */
export function createDogStatsDClient(enabled: boolean, host: string, port: number): DogStatsDClient | null {
	if (!enabled) return null;

	let socket: Socket | undefined;
	let closed = false;

	function ensureSocket(): Socket {
		if (!socket) {
			socket = createSocket("udp4");
			// Required fail-open guard: without an 'error' listener, an async
			// socket error (e.g. ECONNREFUSED from ICMP port-unreachable on Linux
			// when no DogStatsD agent listens) is emitted as an unhandled 'error'
			// event and crashes the agent process. The send callback has already
			// fired by then, so this listener is the only defense.
			socket.on("error", () => {
				// Drop the broken socket; the next metric creates a fresh one.
				try {
					socket?.close();
				} catch {
					// already closed
				}
				socket = undefined;
			});
			socket.unref();
		}
		return socket;
	}

	return {
		send(packet) {
			if (closed) return;
			try {
				ensureSocket().send(Buffer.from(packet, "utf8"), port, host, () => {
					// Fire-and-forget: ignore DNS/send errors, UDP is best-effort.
				});
			} catch {
				// Fail-open: metrics are best-effort.
			}
		},
		close() {
			closed = true;
			try {
				socket?.close();
			} catch {
				// already closed
			}
			socket = undefined;
		},
	};
}

/** Build a sorted tag list, sanitizing values for the DogStatsD wire format. */
function safeTags(values: Record<string, string>): string[] {
	const tags: string[] = [];
	for (const key of Object.keys(values).sort()) {
		const raw = values[key] || "unknown";
		tags.push(`${key}:${String(raw).replace(TAG_SANITIZE_RE, "_")}`);
	}
	return tags;
}

function datadogTokensExtensionImpl(pi: ExtensionAPI, client: DogStatsDClient): void {
	pi.on("message_end", (event) => {
		try {
			const message = event.message;
			if (message.role !== "assistant") return;
			const msg = message as AssistantMessage;

			const tags = safeTags({
				model: msg.responseModel || msg.model,
				provider: msg.provider,
				api: msg.api,
			});
			const tagSuffix = `#${tags.join(",")}`;

			const lines: string[] = [];
			// Counter metrics — skip zero values to avoid empty timeseries.
			// Aborted messages carry partial usage; the repo's usage accounting
			// excludes them, so skip token counters but still count the attempt.
			if (msg.stopReason !== "aborted") {
				const counts: Array<[string, number]> = [
					["prime.tokens.input", msg.usage?.input ?? 0],
					["prime.tokens.output", msg.usage?.output ?? 0],
					["prime.tokens.cache_read", msg.usage?.cacheRead ?? 0],
					["prime.tokens.cache_write", msg.usage?.cacheWrite ?? 0],
					["prime.tokens.total", msg.usage?.totalTokens ?? 0],
				];
				for (const [metric, value] of counts) {
					if (value > 0) {
						lines.push(`${metric}:${value}|c|${tagSuffix}`);
					}
				}
			}

			// Request counter: 1 per assistant message (i.e. per API call).
			lines.push(`prime.api.calls:1|c|${tagSuffix}`);

			// Duration histogram (ms), from request start to message end.
			const durationMs = Date.now() - msg.timestamp;
			if (Number.isFinite(durationMs) && durationMs > 0 && durationMs <= MAX_DURATION_MS) {
				lines.push(`prime.api.duration_ms:${durationMs}|h|${tagSuffix}`);
			}

			// DogStatsD accepts newline-joined multi-metric datagrams; one
			// send per message keeps the datagram rate down.
			client.send(lines.join("\n"));
		} catch {
			// Fail-open: never block or crash the agent loop from a metrics hook.
		}
	});

	// Release the UDP socket when the session's extension runtime tears down
	// (quit, reload, or session replacement) so a long-lived daemon with
	// session churn does not accumulate sockets.
	pi.on("session_shutdown", () => {
		client.close();
	});
}

/**
 * Extension factory for Datadog token metrics. Self-disables when
 * PRIME_AGENT_DATADOG_METRICS is not "1", so it is safe to always load.
 */
export function createDatadogTokensExtension(): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		const enabled = process.env.PRIME_AGENT_DATADOG_METRICS === "1";
		if (!enabled) return;
		const host = process.env.PRIME_AGENT_DATADOG_AGENT_HOST || "127.0.0.1";
		const port = parsePortEnv("PRIME_AGENT_DATADOG_AGENT_PORT", 8125);
		const client = createDogStatsDClient(enabled, host, port);
		if (!client) return;
		datadogTokensExtensionImpl(pi, client);
	};
}
