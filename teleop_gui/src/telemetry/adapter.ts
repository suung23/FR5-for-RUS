import { config, type AppConfig } from './config';
import { HttpPollTransport } from './transports/http';
import { RosbridgeTransport } from './transports/rosbridge';
import { SimulationTransport } from './transports/simulation';
import { WebSocketTransport } from './transports/websocket';
import type {
  LinkStatus,
  RobotTelemetry,
  Transport,
  TransportSink,
  WrenchSample,
} from './types';

export interface AdapterListeners {
  onTelemetry(frame: RobotTelemetry): void;
  onWrench(sample: WrenchSample): void;
  onStatus(status: LinkStatus): void;
}

/**
 * The single seam between the interface and whatever is producing telemetry.
 *
 * Everything above this class works against `RobotTelemetry` and
 * `WrenchSample` and cannot tell which transport is underneath. That is the
 * point: swapping the simulator for a live rosbridge must change where the
 * numbers come from and nothing about what they mean.
 *
 * The adapter **never commands robot motion**. Two things it can send, neither
 * of which moves anything: a calibration command — capture at the pose the
 * operator has already moved to, fit, save — and the operator's station, which
 * only says which way to read a hand motion the operator is already making.
 * Commanding the arm belongs to the ROS 2 stack that owns the velocity clamps
 * and the watchdogs; a monitoring window that could also drive the robot would
 * put a second, unguarded path to the hardware on the same screen as the
 * readouts meant to make it safe.
 *
 * The channel exists because the procedure has to say "capture now" from
 * somewhere, and the operator is at this screen. Its boundary is enforced by
 * what the bridge accepts, not only by what this sends.
 */
export class RobotTelemetryAdapter {
  private transport: Transport | null = null;
  private status: LinkStatus;
  private staleTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly listeners: AdapterListeners,
    cfg: AppConfig = config,
  ) {
    const transport = createTransport(cfg);
    this.transport = transport;
    this.status = {
      phase: 'idle',
      transport: transport.kind,
      endpoint: transport.endpoint,
      attempts: 0,
    };
  }

  get linkStatus(): LinkStatus {
    return this.status;
  }

  start(): void {
    if (!this.transport) return;
    const sink: TransportSink = {
      onTelemetry: (frame) => this.listeners.onTelemetry(frame),
      onWrench: (sample) => this.listeners.onWrench(sample),
      onStatus: (patch) => this.patchStatus(patch),
    };
    this.transport.start(sink);

    // A link can sit "connected" while the far end has gone quiet. Watch the
    // gap since the last frame so the operator sees staleness rather than a
    // frozen pose that still looks live.
    this.staleTimer = setInterval(() => {
      const last = this.status.lastFrameAt;
      if (this.status.phase !== 'connected' || last === undefined) return;
      if (Date.now() - last > 1500) {
        this.patchStatus({ phase: 'error', error: 'no frames received' });
        this.listeners.onTelemetry({ timestamp: Date.now(), connected: false });
      }
    }, 500);
  }

  /**
   * Send a console command. Never a motion command.
   *
   * Returns false when the transport has no back channel — the simulator and
   * the polling transports do not, and the UI has to say so rather than
   * pretending a capture started.
   */
  sendCommand(command: Record<string, unknown>): boolean {
    return this.transport?.sendCommand?.(command) ?? false;
  }

  stop(): void {
    if (this.staleTimer) clearInterval(this.staleTimer);
    this.staleTimer = null;
    this.transport?.stop();
  }

  private patchStatus(patch: Partial<LinkStatus>): void {
    this.status = { ...this.status, ...patch };
    this.listeners.onStatus(this.status);
  }
}

function createTransport(cfg: AppConfig): Transport {
  switch (cfg.transport) {
    case 'websocket':
      return new WebSocketTransport(cfg.telemetryUrl);
    case 'http':
      return new HttpPollTransport(cfg.telemetryUrl, cfg.httpPollMs, cfg.wrenchUrl || undefined);
    case 'rosbridge':
      return new RosbridgeTransport(cfg.telemetryUrl, cfg.rosJointTopic, cfg.rosWrenchTopic);
    case 'simulation':
    default:
      return new SimulationTransport();
  }
}
