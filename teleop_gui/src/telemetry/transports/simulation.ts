import { config } from '../config';
import type {
  ForceWaveform,
  ForceWaveformBin,
  ProbingMode,
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

/** Step period, milliseconds. 50 Hz — motion reads as continuous. */
const SIM_STEP_MS = 20;
/** Waveform bins per step. Two 10 ms bins, matching the bridge's default. */
const SIM_WAVEFORM_BINS = 2;

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
  /** Latched safety level, so the state does not chatter at the threshold. */
  private safety: SafetyState = 'normal';
  /** Velocity-limit mode. One-way, mirroring the control stack. */
  private probing: ProbingMode = 'approach';
  /**
   * Operator station, as the control stack would declare it.
   *
   * Simulated so the station panel is operable without the ROS stack — a
   * control that is permanently dead in the demo teaches the operator it does
   * nothing. The pending-then-applied step is kept because that delay is the
   * part worth rehearsing: the change lands on the next grip, not on the click.
   */
  private operatorYawDeg = 0;
  private pendingYawDeg: number | null = null;
  private gripsLeft = 0;

  /**
   * The 20 ms since the last step, as the bridge would fold it.
   *
   * The real sensor runs at 1 kHz and the bridge sends one frame a second with
   * that second's shape folded into bins; this step is 20 ms and produces the
   * two bins that span it. Sub-samples are generated per bin rather than reused
   * so the envelope has a width for the plot to shade — without that the band
   * would be reviewable only against hardware, which is the one place a
   * display bug is expensive to find.
   */
  private waveform(force: [number, number, number]): ForceWaveform {
    const bins: ForceWaveformBin[] = [];
    for (let i = 0; i < SIM_WAVEFORM_BINS; i += 1) {
      let sum = 0;
      let low = Infinity;
      let high = -Infinity;
      for (let k = 0; k < 10; k += 1) {
        // Sensor-rate noise, an order finer than the per-step figure: what the
        // averaging removes is what the band is there to put back.
        const fz = force[2] + noise(0.05);
        sum += fz;
        low = Math.min(low, fz);
        high = Math.max(high, fz);
      }
      bins.push({
        ageMs: ((SIM_WAVEFORM_BINS - 1 - i) * SIM_STEP_MS) / SIM_WAVEFORM_BINS,
        fx: force[0],
        fy: force[1],
        fz: sum / 10,
        fzMin: low,
        fzMax: high,
      });
    }
    return { hz: (1000 * SIM_WAVEFORM_BINS) / SIM_STEP_MS, bins };
  }

  /**
   * Accept the one command the console can send here.
   *
   * The simulator has no robot to protect, but it keeps the same contract:
   * the change is announced as pending and only takes effect a beat later,
   * standing in for the next grip.
   */
  sendCommand(command: Record<string, unknown>): boolean {
    if (command.command !== 'teleop.operator_frame') return false;
    const yaw = Number(command.yawDeg);
    if (!Number.isFinite(yaw)) return false;
    if (yaw === this.operatorYawDeg) {
      this.pendingYawDeg = null;
      return true;
    }
    this.pendingYawDeg = yaw;
    this.gripsLeft = 12;         // ~12 frames, long enough to read as a wait
    return true;
  }

  start(sink: TransportSink): void {
    this.t0 = Date.now();
    this.phaseStart = this.t0;
    sink.onStatus({ phase: 'connected', attempts: 0, error: undefined });

    // 50 Hz. Fast enough that motion reads as continuous, slow enough that the
    // chart buffer covers a useful span without thinning.
    this.timer = setInterval(() => this.step(sink), SIM_STEP_MS);
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
      // Two episodes, because the display has to be seen doing all of its job.
      //
      //   0-9 s   light coupling. Total F_n lands near 5.5 N: the working band,
      //           below the 6 N warning. This is where a session mostly lives.
      //   9-14 s  a firmer press that crosses 7 N, so the contact stage, the
      //           latch, the amber warning and the red limit are all exercised
      //           rather than sitting unused behind a threshold nothing reaches.
      //
      // Living permanently in protective stop would be just as wrong — it
      // teaches the operator to ignore the colour that matters most.
      const ramp = Math.min(1, held / 1.5);
      // The firm episode has to clear the contact-probing threshold, so it is
      // sized from the configured value rather than a constant that would
      // silently stop exercising the transition when the threshold moves.
      const firmPeak = config.contactProbingN - 3.9 - 1.55 + 1.2;
      const firm = held > 9 ? Math.min(1, (held - 9) / 1.2) * Math.max(0, firmPeak) : 0;
      // Light episode sits at F_n ~5.3 N with a narrow wobble so it stays
      // clearly inside the working band; the firm episode is what crosses.
      this.contactForce =
        ramp * (3.75 + firm + 0.22 * Math.sin(held * 1.9) + 0.09 * Math.sin(held * 5.3));
    } else if (this.phase === 'retract') {
      this.contactForce = Math.max(0, this.contactForce - 0.35);
    } else {
      this.contactForce = 0;
    }

    // Compression is negative F_z under `normal_force_sign = -1`.
    const fz = gravityFz - this.contactForce + noise(0.05);
    const lateral = this.phase === 'contact' ? 0.35 : 0.05;

    const force: [number, number, number] = [
      Math.sin(t * 0.55) * lateral + noise(0.05),
      Math.cos(t * 0.43) * lateral + noise(0.05),
      fz,
    ];

    sink.onWrench({
      timestamp: now,
      source: 'simulation',
      force,
      torque: [
        this.contactForce * 0.004 + noise(0.002),
        this.contactForce * 0.003 + noise(0.002),
        noise(0.002),
      ],
      forceWaveform: this.waveform(force),
    });

    // --- state ------------------------------------------------------------
    // The FR5 controller has no force sensor of its own — the PX6D is on USB —
    // so it can only report that it is being teleoperated, never that it is in
    // contact. Claiming otherwise here would fake an agreement the real system
    // cannot produce.
    const robotState: RobotState = this.phase === 'idle' ? 'idle' : 'teleop';
    // Hysteresis on the declared safety level. A bare comparison flaps every
    // few samples while the force oscillates near a threshold, and each flap
    // would be a line in the event log — which is how a log stops being read.
    //
    // Thresholds come from the configuration, not from constants: a simulator
    // that keeps announcing a protective stop at a limit the console no longer
    // uses teaches the operator to distrust both.
    const fn = -fz;
    const { warnForceN, maxForceN, contactProbingN } = config;
    if (fn >= maxForceN) this.safety = 'protective_stop';
    else if (fn >= warnForceN) {
      if (this.safety !== 'protective_stop') this.safety = 'warning';
    } else if (fn < warnForceN - 1.0) this.safety = 'normal';
    const safetyState: SafetyState = this.safety;

    // The control stack's mode switch is one-way; mirror that here so the
    // console's gate and its VELOCITY LIMITS field are exercised.
    if (fn >= contactProbingN) this.probing = 'contact_probing';

    // Stand-in for the next grip: the requested station lands a beat after it
    // was asked for, never on the click itself.
    if (this.pendingYawDeg !== null && --this.gripsLeft <= 0) {
      this.operatorYawDeg = this.pendingYawDeg;
      this.pendingYawDeg = null;
    }

    sink.onTelemetry({
      timestamp: now,
      connected: true,
      jointPositions: this.joints,
      jointVelocities: this.velocities,
      robotState,
      safetyState,
      probingMode: this.probing,
      teleopFrame: {
        operatorYawDeg: this.operatorYawDeg,
        pendingYawDeg: this.pendingYawDeg,
        mirrored: Math.abs(this.operatorYawDeg) > 90,
        linearFrame: 'latched',
        angularFrame: 'body',
        tipRollDeg: 0,
        engageCount: 1,
      },
    });
    sink.onStatus({ lastFrameAt: now });
  }
}

/** Zero-mean noise with the given approximate amplitude. */
function noise(amplitude: number): number {
  return (Math.random() - 0.5) * 2 * amplitude;
}
