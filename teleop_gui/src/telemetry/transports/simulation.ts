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
  /** Velocity-limit mode, with the control stack's hysteresis. */
  private probing: ProbingMode = 'approach';
  /** Seconds the contact force has been continuously under the release level. */
  private belowReleaseS = 0;
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
  /**
   * Working-pose zero, as the bridge would hold it.
   *
   * Simulated because a Zero button that does nothing without hardware teaches
   * the operator it does nothing at all. The refusals are simulated too: this
   * zero is a constant, so taking it with load on the probe folds that load in
   * and the arm stops reading contact as contact.
   */
  private tare: [number, number, number] = [0, 0, 0];
  /** Selected in-plane mode. Applies at the next contact transition. */
  private inplane = false;

  sendCommand(command: Record<string, unknown>): boolean {
    if (command.command === 'calib.tare') {
      const magnitude = Math.hypot(...this.contactProbe());
      if (magnitude > 2.0) {
        this.ack('calib.tare', false,
          `지금 ${magnitude.toFixed(2)} N 이 실려 있다 — 무언가 닿아 있다`);
        return true;
      }
      const [x, y, z] = this.contactProbe();
      this.tare = [this.tare[0] + x, this.tare[1] + y, this.tare[2] + z];
      this.ack('calib.tare', true);
      return true;
    }
    if (command.command === 'calib.tare.clear') {
      this.tare = [0, 0, 0];
      this.ack('calib.tare.clear', true);
      return true;
    }
    if (command.command === 'contact.inplane') {
      // Accepted during approach too — arming it beforehand is the normal way
      // to use it, because contact starts without warning.
      this.inplane = command.enabled === true;
      if (this.probing !== 'approach') {
        this.probing = this.inplane ? 'contact_probing_inplane' : 'contact_probing';
      }
      this.ack('contact.inplane', true);
      return true;
    }
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

  /** The compensated contact-point force before the working zero. */
  private contactProbe(): [number, number, number] {
    return [this.lateralX, this.lateralY, -this.contactForce];
  }

  private ack(command: string, ok: boolean, reason?: string): void {
    // Answered on the next tick, as a socket would: a button that reports back
    // before the frame it changed has arrived teaches the wrong cadence.
    setTimeout(() => this.sink?.onAck({ command, ok, reason }), 20);
  }

  private sink: TransportSink | null = null;
  private lateralX = 0;
  private lateralY = 0;

  start(sink: TransportSink): void {
    this.sink = sink;
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
      //   0-9 s   the hold band. This is where a session mostly lives, and it
      //           is where the robot's own regulator would be holding it.
      //   9-14 s  a firmer press that crosses warn and limit, so the amber
      //           warning and the red outline are exercised rather than sitting
      //           unused behind a threshold nothing reaches.
      //
      // Living permanently in protective stop would be just as wrong — it
      // teaches the operator to ignore the colour that matters most.
      //
      // Both levels are sized from the configuration rather than from constants.
      // The thresholds have moved twice (contact 8 → 1 N, hold 5 → 3 N) and each
      // time a hard-coded episode would have quietly stopped crossing the line
      // it existed to cross — which is the failure a simulator cannot report,
      // because it looks exactly like an interface that has nothing to show.
      const ramp = Math.min(1, held / 1.5);
      const firmPeak = Math.max(0, config.maxForceN + 0.5 - config.targetForceN);
      const firm = held > 9 ? Math.min(1, (held - 9) / 1.2) * firmPeak : 0;
      this.contactForce =
        ramp *
        (config.targetForceN + firm + 0.22 * Math.sin(held * 1.9) + 0.09 * Math.sin(held * 5.3));
    } else if (this.phase === 'retract') {
      this.contactForce = Math.max(0, this.contactForce - 0.35);
    } else {
      this.contactForce = 0;
    }

    // Compression is negative F_z under `normal_force_sign = -1`.
    const fz = gravityFz - this.contactForce + noise(0.05);
    const lateral = this.phase === 'contact' ? 0.35 : 0.05;

    this.lateralX = Math.sin(t * 0.55) * lateral + noise(0.05);
    this.lateralY = Math.cos(t * 0.43) * lateral + noise(0.05);
    const force: [number, number, number] = [this.lateralX, this.lateralY, fz];

    // The compensated contact-point wrench: the same contact, with the tool's
    // own weight taken out. Emitted because the console's stage machine now
    // judges on `‖F‖` at a 1 N threshold, and refuses to judge at all without a
    // calibration — so a simulator that never declared one would leave the
    // stage, the latch and the timeline permanently unexercised, which is the
    // half of the interface most worth reviewing before hardware.
    // The working zero comes off here, upstream of everything the console
    // reads — exactly where the bridge subtracts its own working tare, so the
    // plot, the plate and the stage classifier all move together.
    const contactProbe: [number, number, number] = [
      force[0] - this.tare[0],
      force[1] - this.tare[1],
      -this.contactForce + noise(0.02) - this.tare[2],
    ];
    const torque: [number, number, number] = [
      this.contactForce * 0.004 + noise(0.002),
      this.contactForce * 0.003 + noise(0.002),
      noise(0.002),
    ];

    sink.onWrench({
      timestamp: now,
      source: 'simulation',
      force,
      torque,
      forceWaveform: this.waveform(force),
      calibrationValid: true,
      calibrationIssues: [],
      compensated: {
        rawSensor: [...force, ...torque],
        biasCorrectedSensor: [...force, ...torque],
        externalSensor: [...contactProbe, ...torque],
        externalProbe: [...contactProbe, ...torque],
        contactProbe: [...contactProbe, ...torque],
        normalForceN: contactProbe[2],
      },
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

    // The mode switch, with the hysteresis the control stack gained on
    // 2026-08-31 — enter on ‖F‖, leave once it has stayed under the release
    // level for the confirmation window.
    //
    // Mirroring the release matters more than it sounds. While this was modelled
    // as one-way, a threshold of 1 N meant the simulated console latched into
    // contact probing seconds after starting and stayed there for the rest of
    // the session — so the palette, the standing notice and the VELOCITY LIMITS
    // field were all permanently on, and the transition they exist to show was
    // visible exactly once, before anyone had looked.
    // Judged on the **compensated** contact-point wrench, which is what
    // `us_diff_ik` subscribes to. The raw channels still carry the tool's own
    // weight — about 1.5 N here, above the 1 N threshold — so a simulator that
    // switched on those would declare contact while hanging in free space, and
    // would be reproducing a bug the real stack fixed rather than the stack.
    const magnitude = Math.hypot(contactProbe[0], contactProbe[1], contactProbe[2]);
    if (this.probing === 'approach') {
      if (magnitude >= contactProbingN) {
        this.probing = this.inplane ? 'contact_probing_inplane' : 'contact_probing';
        this.belowReleaseS = 0;
      }
    } else if (magnitude <= config.contactProbingReleaseN) {
      this.belowReleaseS += 0.02;
      if (this.belowReleaseS >= config.contactProbingReleaseS) {
        this.probing = 'approach';
      }
    } else {
      this.belowReleaseS = 0;
    }

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
      inplaneRotation: this.inplane,
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
