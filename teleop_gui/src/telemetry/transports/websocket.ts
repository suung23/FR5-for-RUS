import type { Transport, TransportSink } from '../types';
import { parseTelemetry, parseWrench } from './parse';

type FrameKind = 'telemetry' | 'wrench' | 'unknown';

/** Declared frame type, or an inference from the payload's shape. */
function frameKind(payload: unknown): FrameKind {
  if (payload && typeof payload === 'object') {
    const declared = (payload as Record<string, unknown>).type;
    if (declared === 'telemetry' || declared === 'wrench') return declared;
    if ('force' in (payload as object) && !('jointPositions' in (payload as object))) {
      return 'wrench';
    }
  }
  return 'unknown';
}

/**
 * Plain websocket transport.
 *
 * Expects a server that pushes JSON frames matching `RobotTelemetry`, and
 * optionally wrench frames on the same socket (distinguished by a `force`
 * field). Reconnects with capped exponential backoff; it never gives up
 * silently, because a monitor that stops trying looks identical to a robot
 * that stopped moving.
 */
export class WebSocketTransport implements Transport {
  readonly kind = 'websocket' as const;
  private socket: WebSocket | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;
  private stopped = false;
  private sink: TransportSink | null = null;

  constructor(readonly endpoint: string) {}

  start(sink: TransportSink): void {
    this.sink = sink;
    this.stopped = false;
    this.open();
  }

  /** Console commands only, never motion — see `Transport.sendCommand`. */
  sendCommand(command: Record<string, unknown>): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(command));
    return true;
  }

  stop(): void {
    this.stopped = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.socket?.close();
    this.socket = null;
    this.sink?.onStatus({ phase: 'closed' });
  }

  private open(): void {
    if (this.stopped || !this.sink) return;
    const sink = this.sink;
    sink.onStatus({
      phase: this.attempts === 0 ? 'connecting' : 'reconnecting',
      attempts: this.attempts,
    });

    let socket: WebSocket;
    try {
      socket = new WebSocket(this.endpoint);
    } catch (err) {
      this.fail(err);
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.attempts = 0;
      sink.onStatus({ phase: 'connected', attempts: 0, error: undefined });
    };

    socket.onmessage = (event) => {
      const now = Date.now();
      let payload: unknown;
      try {
        payload = JSON.parse(String(event.data));
      } catch {
        return; // A single bad frame is not worth tearing the link down.
      }
      // One socket carries two kinds of frame. Dispatch on the declared type,
      // and fall back to the shape only for servers that do not tag.
      //
      // Guessing from shape alone is what went wrong here first: a wrench
      // frame has no joint fields, so parsing it as telemetry too produced a
      // valid-looking frame with everything absent, and at 50 Hz that wiped
      // the joint state between every telemetry update.
      const kind = frameKind(payload);

      if (kind !== 'telemetry') {
        const wrench = parseWrench(payload, now);
        if (wrench) {
          sink.onWrench(wrench);
          sink.onStatus({ lastFrameAt: now });
          if (kind === 'wrench') return;
        }
      }

      if (kind !== 'wrench') {
        const frame = parseTelemetry(payload, now);
        // Do NOT force `connected: true` here. The socket being open says the
        // bridge is reachable, not that the robot is. The bridge reports
        // `connected: false` when joint state has gone stale, and overriding
        // that made the console show ROBOT CONNECTED with no robot behind it —
        // the exact failure this display exists to prevent.
        if (frame) sink.onTelemetry(frame);
      }
      sink.onStatus({ lastFrameAt: now });
    };

    socket.onerror = () => this.fail(new Error('websocket error'));
    socket.onclose = () => {
      if (this.stopped) return;
      sink.onTelemetry({ timestamp: Date.now(), connected: false });
      this.scheduleRetry();
    };
  }

  private fail(err: unknown): void {
    this.sink?.onStatus({
      phase: 'error',
      error: err instanceof Error ? err.message : String(err),
    });
    this.scheduleRetry();
  }

  private scheduleRetry(): void {
    if (this.stopped || this.retryTimer) return;
    this.attempts += 1;
    const delay = Math.min(500 * 2 ** Math.min(this.attempts, 5), 10_000);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.open();
    }, delay);
  }
}
