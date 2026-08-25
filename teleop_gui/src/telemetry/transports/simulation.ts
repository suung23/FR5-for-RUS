import type {
  RobotState,
  SafetyState,
  Transport,
  TransportSink,
} from '../types';

/**
 * Built-in simulator — the default transport.
 *
 * This exists so the interface can be run, reviewed and demonstrated without
 * opening a socket against lab hardware, and so every panel has something
 * honest to render. It is clearly labelled as simulated throughout the UI; it
 * is never a stand-in for a measurement.
 *
 * The numbers it produces are shaped by this cell's real characteristics
 * rather than invented:
 *
 * - Force noise is about +/-0.05 N per axis, matching the PX6D's 0.1 %FS spec
 *   and the bench measurement of 2026-08-24.
 * - Unloaded `F_z` sits near -1.4 N. That is the probe/tool weight projected
 *   onto the sensor z axis, and it drifts with pose because gravity is not
 *   compensated yet (payload identification is still pending).
 * - Contact rises to roughly 5-6 N, below the 7 N hard limit, then holds with
 *   the small oscillation a hand-held force setpoint actually shows.
 * - Joint motion stays inside `probe.yaml` joint limits and moves at freespace
 *   speeds during approach, contact speeds once touching.
 */

type Phase = 'idle' | 'approach' | 'contact' | 'retract';

const HOME = [0, -1.2, 1.4, -1.75, -1.57, 0];
const OVER_TARGET = [0.35, -1.05, 1.55, -2.05, -1.57, 0.35];

export class SimulationTransport implements Transport {
  readonly kind = 'simulation' as const;
  readonly endpoint = 'built-in simulator (no network traffic)';

  private timer: ReturnType<typeof setInterval> | null = null;
  private t0 = 0;
  private phase: Phase = 'idle';
  private phaseStart = 0;
  private joints = [...HOME];
  private velocities = new Array(6).fill(0);
  private contactForce = 0;

  start(sink: TransportSink): void {
    this.t0 = Date.now();
    this.phaseStart = this.t0;
    sink.onStatus({ phase: 'connected', attempts: 0, error: undefined });

    // 50 Hz. Fast enough that motion reads as continuous, slow enough that the
    // chart buffer covers a useful span without thinning.
    this.timer = setInterval(() => this.step(sink), 20);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private advancePhase(now: number): void {
    const held = (now - this.phaseStart) / 1000;
    const next: Record<Phase, [Phase, number]> = {
      idle: ['approach', 3],
      approach: ['contact', 7],
      contact: ['retract', 14],
      retract: ['idle', 4],
    };
    const [to, after] = next[this.phase];
    if (held > after) {
      this.phase = to;
      this.phaseStart = now;
    }
  }

  private step(sink: TransportSink): void {
    const now = Date.now();
    this.advancePhase(now);
    const t = (now - this.t0) / 1000;
    const held = (now - this.phaseStart) / 1000;

    // --- joints -----------------------------------------------------------
    const target =
      this.phase === 'idle' || this.phase === 'retract' ? HOME : OVER_TARGET;
    // Contact motion is deliberately an order slower, mirroring the
    // freespace (150 mm/s) versus contact (10 mm/s) clamp split.
    const gain = this.phase === 'contact' ? 0.012 : 0.06;

    const previous = [...this.joints];
    this.joints = this.joints.map((q, i) => {
      const drift =
        this.phase === 'contact' ? Math.sin(t * 0.7 + i) * 0.004 : 0;
      return q + (target[i] - q) * gain + drift;
    });
    this.velocities = this.joints.map((q, i) => (q - previous[i]) / 0.02);

    // --- wrench -----------------------------------------------------------
    // Gravity projection onto sensor z, varying with wrist pose. This is the
    // uncompensated payload term the UI warns about.
    const gravityFz = -1.4 * Math.cos(this.joints[4] + 1.57) - 0.15;

    if (this.phase === 'contact') {
      // Rise, then hold near 5.5 N with the wobble of a hand-held setpoint.
      const ramp = Math.min(1, held / 1.5);
      this.contactForce = ramp * (5.5 + 0.55 * Math.sin(held * 1.9) + 0.2 * Math.sin(held * 5.3));
    } else if (this.phase === 'retract') {
      this.contactForce = Math.max(0, this.contactForce - 0.35);
    } else {
      this.contactForce = 0;
    }

    // Compression is negative F_z under `normal_force_sign = -1`.
    const fz = gravityFz - this.contactForce + noise(0.05);
    const lateral = this.phase === 'contact' ? 0.35 : 0.05;

    sink.onWrench({
      timestamp: now,
      force: [
        Math.sin(t * 0.55) * lateral + noise(0.05),
        Math.cos(t * 0.43) * lateral + noise(0.05),
        fz,
      ],
      torque: [
        this.contactForce * 0.004 + noise(0.002),
        this.contactForce * 0.003 + noise(0.002),
        noise(0.002),
      ],
    });

    // --- state ------------------------------------------------------------
    const robotState: RobotState =
      this.phase === 'idle' ? 'idle' : this.phase === 'contact' ? 'contact' : 'teleop';
    const fn = -fz;
    const safetyState: SafetyState =
      fn >= 7 ? 'protective_stop' : fn >= 6 ? 'warning' : 'normal';

    sink.onTelemetry({
      timestamp: now,
      connected: true,
      jointPositions: this.joints,
      jointVelocities: this.velocities,
      robotState,
      safetyState,
    });
    sink.onStatus({ lastFrameAt: now });
  }
}

/** Zero-mean noise with the given approximate amplitude. */
function noise(amplitude: number): number {
  return (Math.random() - 0.5) * 2 * amplitude;
}
