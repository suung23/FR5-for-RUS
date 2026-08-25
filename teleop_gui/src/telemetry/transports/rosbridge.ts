import type { Transport, TransportSink } from '../types';
import { parseWrench } from './parse';
import type { RobotTelemetry } from '../types';

/**
 * rosbridge_suite transport (`rosbridge_websocket`, default port 9090).
 *
 * Subscribes to a JointState topic and a WrenchStamped topic and translates
 * them into the shared contract. This is the transport that fits the existing
 * ROS 2 stack: `us_servo_node` already publishes joint state, and a PX6D
 * publisher would sit on the wrench topic.
 *
 * ROS 2 timestamps arrive as `{sec, nanosec}` in the message header. They are
 * converted here rather than in the parser because only this transport knows
 * the message shape.
 */
export class RosbridgeTransport implements Transport {
  readonly kind = 'rosbridge' as const;
  private socket: WebSocket | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;
  private stopped = false;
  private sink: TransportSink | null = null;
  private latest: RobotTelemetry = { timestamp: 0, connected: false };

  constructor(
    readonly endpoint: string,
    private readonly jointTopic: string,
    private readonly wrenchTopic: string,
  ) {}

  start(sink: TransportSink): void {
    this.sink = sink;
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.send({ op: 'unsubscribe', topic: this.jointTopic });
      this.send({ op: 'unsubscribe', topic: this.wrenchTopic });
    }
    this.socket?.close();
    this.socket = null;
    this.sink?.onStatus({ phase: 'closed' });
  }

  private send(msg: unknown): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(msg));
    }
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
      // throttle_rate caps the bridge's send rate; the PX6D publishes at
      // 1 kHz and no display needs that, but the cap belongs on the wire
      // rather than in the renderer where it would arrive as dropped frames.
      this.send({
        op: 'subscribe',
        topic: this.jointTopic,
        type: 'sensor_msgs/msg/JointState',
        throttle_rate: 20,
      });
      this.send({
        op: 'subscribe',
        topic: this.wrenchTopic,
        type: 'geometry_msgs/msg/WrenchStamped',
        throttle_rate: 20,
      });
    };

    socket.onmessage = (event) => {
      const now = Date.now();
      let envelope: { op?: string; topic?: string; msg?: Record<string, unknown> };
      try {
        envelope = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (envelope.op !== 'publish' || !envelope.msg) return;

      if (envelope.topic === this.jointTopic) {
        const msg = envelope.msg;
        const stamp = rosStampMs(msg.header) ?? now;
        this.latest = {
          ...this.latest,
          timestamp: stamp,
          connected: true,
          jointPositions: finiteArray(msg.position),
          jointVelocities: finiteArray(msg.velocity),
        };
        sink.onTelemetry(this.latest);
      } else if (envelope.topic === this.wrenchTopic) {
        const sample = parseWrench(envelope.msg, now);
        if (sample) sink.onWrench(sample);
      }
      sink.onStatus({ lastFrameAt: now });
    };

    socket.onerror = () => this.fail(new Error('rosbridge socket error'));
    socket.onclose = () => {
      if (this.stopped) return;
      this.latest = { timestamp: Date.now(), connected: false };
      sink.onTelemetry(this.latest);
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

function rosStampMs(header: unknown): number | undefined {
  if (!header || typeof header !== 'object') return undefined;
  const stamp = (header as Record<string, unknown>).stamp;
  if (!stamp || typeof stamp !== 'object') return undefined;
  const { sec, nanosec } = stamp as { sec?: unknown; nanosec?: unknown };
  if (typeof sec !== 'number') return undefined;
  return sec * 1000 + (typeof nanosec === 'number' ? nanosec / 1e6 : 0);
}

function finiteArray(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined;
  const out = value.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  return out.length === value.length ? out : undefined;
}
