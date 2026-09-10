import { create } from 'zustand';
import { RobotTelemetryAdapter } from '../telemetry/adapter';
import { config } from '../telemetry/config';
import { solveChain } from '../telemetry/fr5Model';
import { ContactDetector, type ContactSnapshot } from '../telemetry/contactState';
import type { LinkStatus, RobotTelemetry, UltrasoundFrame, WrenchSample } from '../telemetry/types';
import { logEvent, TransitionWatcher } from './events';

/** One point in the rolling chart buffer. */
export interface ForcePoint {
  t: number;
  /**
   * Raw sensor channels, uncompensated.
   *
   * Kept as the sensor reported them because the components view is the
   * diagnostic for the axis assignment, and that check needs the raw reading.
   */
  fx: number;
  fy: number;
  fz: number;
  /**
   * Contact-point force in the probe frame, `+z` compressing — the same vector
   * and convention the headline plate reads.
   *
   * Stored as a **vector, not a magnitude**, because the operator's display
   * zero is subtracted as a vector. Folding to `‖F‖` here would leave the plot
   * unable to apply the zero the plate applies, and the two would disagree
   * about the same contact by the size of the offset.
   */
  cx: number;
  cy: number;
  cz: number;
  /**
   * Normal channel `[min, max]` over the span this point covers.
   *
   * A point drawn from a waveform bin stands for ten milliseconds of a 1 kHz
   * stream, so it has a width as well as a value. Only the normal channel has
   * per-bin extremes to work from — the bridge sends that envelope, not the
   * lateral pair — so the band is the magnitude evaluated at these two ends
   * with the bin's own mean lateral load. Equal to `cz` when the frame carried
   * a single reading and there is no width to show.
   */
  czLo: number;
  czHi: number;
}

/**
 * The contact-force magnitude a point stands for, signed by whether the probe
 * is pressing.
 *
 * No zero is applied here, and none should be. The operator's zero is the
 * bridge's working tare (`calib.tare`), subtracted inside `compensate` before
 * the wrench is published — so everything downstream, this console and the
 * robot's own mode switch alike, is already reading from the same zeroed
 * value. A second offset applied here would only put the two back out of step.
 */
export function contactMagnitude(point: ForcePoint, normal: number = point.cz): number {
  const magnitude = Math.hypot(point.cx, point.cy, normal);
  return normal >= 0 ? magnitude : -magnitude;
}

/** Seconds of force history retained for the plots. */
const HISTORY_SECONDS = 20;
/** Seconds of flange path drawn in the workspace. */
const TRAJECTORY_SECONDS = 10;
/** Path sample rate. Denser than this and the polyline is noise, not a path. */
const TRAJECTORY_HZ = 10;
/**
 * Rate at which the chart buffer is handed to React.
 *
 * Not the resolution of the trace — a waveform frame contributes all of its
 * bins at once, so the drawn line can be far finer than this. What this bounds
 * is how often the plot is asked to redraw, which is a frame-budget question
 * and nothing to do with how much of the signal is kept.
 *
 * 🔁 2026-08-31: 25 → 5, **deliberately below** the bridge's frame rate
 * (`bridge.wrench_stream_hz`, 10). The earlier note here said it had to stay
 * above that rate or frames would be held back and the trace would advance in
 * slabs. Frames are indeed held back now — but nothing is lost, because
 * `pendingPoints` keeps every bin until the flush, and a slab on a twenty
 * second axis is 200 ms, one percent of the width.
 *
 * The reason to pay that is repaint cost. A CPU profile of the running console
 * came back 77% idle, so the plot is not spending main-thread time; what it
 * spends is rasterisation of two long SVG paths plus a filled band, and that
 * scales directly with how often they are repainted. Halving the repaints is
 * the one lever that always works on that.
 */
const CHART_HZ = 5;

interface TelemetryState {
  telemetry: RobotTelemetry;
  wrench: WrenchSample | null;
  /** Latest ultrasound picture, or null before the first one arrives. */
  ultrasound: UltrasoundFrame | null;
  contact: ContactSnapshot;
  /**
   * Contact-point force `[Fx, Fy, Fz]` in the probe frame, `+z` compressing.
   *
   * Derived once, here, so the headline reading, the stage classifier, the
   * peak hold and the trend cannot end up describing different quantities.
   * Null until a wrench arrives.
   */
  contactForce: [number, number, number] | null;
  /**
   * Whether the stage above is a judgement at all.
   *
   * False while no calibration is loaded, mirroring `us_diff_ik`'s
   * `require_calibration` gate. Before compensation the wrench still carries
   * the mount and probe weighing themselves — about 10 N measured — and that
   * offset changes sign with pose, so `‖F‖` clears the 1 N threshold with
   * nothing touching the probe. The control stack refuses to classify on a
   * number it cannot trust; a console that classified anyway would sit at
   * CONTACT for the whole session and teach the operator to ignore the field.
   */
  contactJudged: boolean;
  link: LinkStatus | null;
  history: ForcePoint[];
  /**
   * Largest contact-force magnitude ‖F‖ seen since the last operator reset.
   *
   * The same scalar the headline reads and the control stack switches on. Held
   * as an upper bound when the frame carries per-axis window extremes, because
   * the largest value on each axis need not have occurred in the same sample —
   * a peak-hold may over-report, never under-report.
   */
  peakContactForceN: number;
  /** Frames received per second, measured. */
  frameRateHz: number;
  /**
   * Bin rate of the waveform the plots are drawing, or null when the source
   * sends a single reading per frame and the trace is only as fine as the
   * frame rate. The chart says which of the two it is showing.
   */
  waveformHz: number | null;
  /** Recent flange positions in metres, oldest first, for the workspace path. */
  trajectory: [number, number, number][];
  paused: boolean;

  setPaused(paused: boolean): void;
  resetSession(): void;
  /** Send a console command. Returns false when there is no back channel. */
  sendCommand(command: Record<string, unknown>): boolean;
}

/**
 * The contact-point force, in the sensor's sign convention.
 *
 * `probe.yaml` defines the normal force as `F_n = normal_force_sign · F_z^probe`,
 * and `us_diff_ik` closes its regulator on exactly that. So the probe-frame `z`
 * that comes back here is **negative under compression**, the same way the raw
 * channel is, and the sign is applied once at the display boundary rather than
 * assumed to have happened already.
 *
 * With a calibration loaded this is the compensated contact-point wrench — the
 * tool's own weight removed, rotated into the probe frame, moment reference
 * moved to the tip. Without one it falls back to the raw channels, which still
 * carry the payload. The console must read whichever of the two the **robot**
 * is regulating, because the bridge publishes the compensated wrench to
 * `wrench_px6d` the moment a profile exists; a console still reading the raw
 * channel would then disagree with the arm about how hard it is pressing.
 */
function contactPointForce(sample: WrenchSample): [number, number, number] {
  const stages = sample.compensated;
  if (!stages) return [sample.force[0], sample.force[1], sample.force[2]];
  return [stages.contactProbe[0], stages.contactProbe[1], stages.contactProbe[2]];
}

/**
 * The scalar the control stack acts on, mirroring `us_diff_ik._control_force`.
 *
 * `ft_sensor.contact_force_mode` is `magnitude`, so this is `‖F‖` carrying the
 * sign of the normal component: positive while the probe is pressing, negative
 * while it is being pulled. The magnitude is what crosses the 1 N threshold —
 * at that force a probe touching even slightly off-axis puts most of the
 * contact into shear, and the normal component alone would read it as no
 * contact at all.
 *
 * @param force Contact-point force in the sensor's sign convention.
 */
function controlForce(force: [number, number, number]): number {
  const magnitude = Math.hypot(force[0], force[1], force[2]);
  return config.normalForceSign * force[2] >= 0 ? magnitude : -magnitude;
}

/** Which subsystem a console command belongs to, for the event log. */
function commandTag(command: string): string {
  if (command.startsWith('teleop.')) return 'TELEOP';
  if (command.startsWith('contact.')) return 'STAGE';
  return 'CALIB';
}

const detector = new ContactDetector({
  enterN: config.contactEnterN,
  releaseN: config.contactReleaseN,
});

const watcher = new TransitionWatcher();
let lastWrenchAt: number | null = null;
let lastChartAt = 0;
/** When the buffer was last handed to React. */
let lastChartFlushAt = 0;
/** Points accepted but not yet handed to React. */
let pendingPoints: ForcePoint[] = [];
/** Last timestamp written to the buffer, so the trace cannot walk backwards. */
let lastPointAt = 0;
let lastPathAt = 0;
const pathStamps: number[] = [];
const frameStamps: number[] = [];

export const useTelemetryStore = create<TelemetryState>((set) => ({
  telemetry: { timestamp: 0, connected: false },
  wrench: null,
  ultrasound: null,
  contact: detector.snapshot(),
  contactForce: null,
  contactJudged: false,
  link: null,
  history: [],
  peakContactForceN: 0,
  frameRateHz: 0,
  waveformHz: null,
  trajectory: [],
  paused: false,

  setPaused: (paused) => set({ paused }),

  sendCommand: (command) => {
    const sent = adapter.sendCommand(command);
    // The tag names the subsystem the command belongs to. A teleoperation
    // change filed under CALIB would be unfindable in the log afterwards.
    const tag = commandTag(String(command.command ?? ''));
    logEvent(tag, sent ? String(command.command) : `${command.command} — 전송 불가`,
             sent ? 'info' : 'warn');
    return sent;
  },

  resetSession: () => {
    detector.reset(true);
    watcher.reset();
    logEvent(
      'SESSION',
      'operator reset — history, peak hold, contact latch and display zero cleared',
    );
    pendingPoints = [];
    lastPointAt = 0;
    set({
      history: [],
      trajectory: [],
      peakContactForceN: 0,
          contact: detector.snapshot(),
    });
  },
}));

const adapter = new RobotTelemetryAdapter({
  onTelemetry(frame) {
    const now = Date.now();
    frameStamps.push(now);
    while (frameStamps.length > 0 && now - frameStamps[0] > 1000) frameStamps.shift();

    if (useTelemetryStore.getState().paused) return;
    watcher.observeSafety(frame.safetyState);
    watcher.observeRobot(frame.robotState);
    watcher.observeProbingMode(frame.probingMode);

    const patch: Partial<TelemetryState> = {
      telemetry: frame,
      frameRateHz: frameStamps.length,
    };

    // The flange path is derived here rather than in the view so the workspace
    // stays a renderer. Sampled well below the frame rate — a 50 Hz polyline
    // is a thick smear, not a trajectory.
    if (frame.jointPositions && now - lastPathAt >= 1000 / TRAJECTORY_HZ) {
      lastPathAt = now;
      const poses = solveChain(frame.jointPositions);
      const tip = poses[poses.length - 1].position;
      const state = useTelemetryStore.getState();
      const next = [...state.trajectory, [tip.x, tip.y, tip.z] as [number, number, number]];
      pathStamps.push(now);
      while (pathStamps.length > 0 && now - pathStamps[0] > TRAJECTORY_SECONDS * 1000) {
        pathStamps.shift();
        next.shift();
      }
      patch.trajectory = next;
    }

    useTelemetryStore.setState(patch);
  },

  onUltrasound(frame) {
    // HOLD freezes the display, and the picture is display. The stream keeps
    // arriving; only what is drawn is held — same rule as the other panels.
    if (useTelemetryStore.getState().paused) return;
    useTelemetryStore.setState({ ultrasound: frame });
  },

  onWrench(sample) {
    const state = useTelemetryStore.getState();
    if (state.paused) return;

    // Contact classification runs on the untared contact force. It is a
    // physical judgement, not a display preference, so it must not follow
    // anything the operator changed about how the plate is drawn — but it must
    // follow the same signal the robot is acting on, which is the compensated
    // one wherever a calibration exists.
    const measured = contactPointForce(sample);
    const dtMs = lastWrenchAt === null ? 0 : sample.timestamp - lastWrenchAt;
    lastWrenchAt = sample.timestamp;

    // No calibration, no judgement — the same gate `us_diff_ik` applies before
    // it will let a contact decision change anything. The reading is still
    // shown, and still labelled RAW; what is withheld is the claim that it
    // means contact.
    const judged = sample.calibrationValid === true;
    const contact = judged
      ? detector.update(controlForce(measured), dtMs)
      : detector.snapshot();
    if (judged) {
      watcher.observeStage(contact.phase, contact.hasContacted, contact.contactForceN);
    }

    // How far compensation moved this frame's reading. Over one window the
    // pose is fixed, so compensation is an affine map and the shift is a
    // constant — which is what lets it be carried onto the waveform bins and
    // the window extremes, neither of which the bridge compensates.
    //
    // On this cell the shift is *exact* on the z channel: the probe frame is a
    // pure rotation about z (mounting 43 deg, no axial flip), so the normal
    // channel is untouched by the rotation and only the payload term moves it.
    const shift: [number, number, number] = [
      measured[0] - sample.force[0],
      measured[1] - sample.force[1],
      measured[2] - sample.force[2],
    ];
    const normalShift = shift[2];

    // The displayed reading is the mean of its window; the peak must not be.
    // Over a one-second readout window a spike inside it would be averaged away,
    // and the peak-hold exists to catch exactly that. The bridge sends the
    // window's own extremes alongside the mean, so the hold sees them.
    //
    // The hold tracks the **total**, because that is what the headline shows
    // and what the robot switches on (2026-08-31). A peak in a different
    // quantity from the reading it sits beside is worse than no peak: it can
    // read *below* the live value and look like a stuck number.
    //
    // Per-axis extremes bound the total but do not give it — the largest |Fx|
    // and the largest |Fy| in a window need not have happened in the same
    // sample. Combining them is therefore an upper bound, never an
    // under-report, which is the direction a peak-hold must err in.
    //
    // Every channel takes the compensation shift first. The extremes arrive as
    // raw sensor channels, and the raw lateral pair carries the sensor's own
    // bias — a peak built from those reads over a newton with nothing touching
    // the probe, which is not an upper bound, it is a wrong number.
    const windowPeak = sample.forceExtremes
      ? Math.hypot(
          Math.max(
            Math.abs(sample.forceExtremes[0][0] + shift[0]),
            Math.abs(sample.forceExtremes[1][0] + shift[0]),
          ),
          Math.max(
            Math.abs(sample.forceExtremes[0][1] + shift[1]),
            Math.abs(sample.forceExtremes[1][1] + shift[1]),
          ),
          Math.max(
            Math.abs(sample.forceExtremes[0][2] + shift[2]),
            Math.abs(sample.forceExtremes[1][2] + shift[2]),
          ),
        )
      : Math.abs(controlForce(measured));

    const patch: Partial<TelemetryState> = {
      wrench: sample,
      contact,
      contactForce: [
        measured[0],
        measured[1],
        config.normalForceSign * measured[2],
      ],
      contactJudged: judged,
      peakContactForceN: Math.max(state.peakContactForceN, windowPeak),
    };

    // Into the chart buffer. Which path runs is decided by what arrived, not by
    // a setting here.
    //
    // A frame carrying a waveform already contains the shape of the window it
    // covers, folded into bins by the bridge — the readout it also carries is a
    // one-second mean because that is what a number on a screen can be, but the
    // line is drawn from the bins and so keeps the sensor's own detail. Without
    // a waveform the frame is one reading, and the old decimation applies: a
    // 50 Hz source has nothing to say between pixels.
    const waveform = sample.forceWaveform;
    if (waveform) {
      for (const bin of waveform.bins) {
        // Every channel takes the compensation shift, not just z. The bins
        // arrive as raw sensor channels, and the raw lateral pair carries the
        // sensor's own bias — a magnitude built from those reads over a newton
        // with nothing touching the probe.
        //
        // The lateral shift is a per-frame constant rather than an exact
        // per-sample correction: within one window the pose is fixed, so
        // compensation is affine and the bias term is removed exactly. What is
        // left is a second-order term from the rotation acting on the signal's
        // own variation inside the window.
        const lo = config.normalForceSign * (bin.fzMin + normalShift);
        const hi = config.normalForceSign * (bin.fzMax + normalShift);
        // Ages are relative to the frame, and the frame is stamped on arrival,
        // so network jitter could otherwise place a bin behind its predecessor.
        const t = Math.max(lastPointAt + 1, sample.timestamp - bin.ageMs);
        lastPointAt = t;
        pendingPoints.push({
          t,
          fx: bin.fx,
          fy: bin.fy,
          fz: bin.fz,
          cx: bin.fx + shift[0],
          cy: bin.fy + shift[1],
          cz: config.normalForceSign * (bin.fz + normalShift),
          czLo: Math.min(lo, hi),
          czHi: Math.max(lo, hi),
        });
      }
    } else if (sample.timestamp - lastChartAt >= 1000 / CHART_HZ) {
      lastChartAt = sample.timestamp;
      lastPointAt = Math.max(lastPointAt + 1, sample.timestamp);
      pendingPoints.push({
        t: lastPointAt,
        fx: sample.force[0],
        fy: sample.force[1],
        fz: sample.force[2],
        cx: measured[0],
        cy: measured[1],
        cz: config.normalForceSign * measured[2],
        czLo: config.normalForceSign * measured[2],
        czHi: config.normalForceSign * measured[2],
      });
    }

    // Hand the buffer over no faster than the plot can use it. Ingestion and
    // redraw are separate rates: a waveform frame adds a hundred points at
    // once, and none of them are lost by waiting for the next flush.
    if (pendingPoints.length > 0 && sample.timestamp - lastChartFlushAt >= 1000 / CHART_HZ) {
      lastChartFlushAt = sample.timestamp;
      const cutoff = lastPointAt - HISTORY_SECONDS * 1000;
      patch.history = state.history.filter((p) => p.t >= cutoff).concat(pendingPoints);
      patch.waveformHz = waveform ? waveform.hz : null;
      pendingPoints = [];
    }

    useTelemetryStore.setState(patch);
  },

  /**
   * The bridge's answer to a console command, put where the operator reads.
   *
   * A calibration command is refused far more often than it succeeds — the
   * probe is not pointing down, something is resting on it, the pose has not
   * arrived — and the reason is the whole content of the answer. Before this
   * the ack was dropped and the button simply appeared not to work.
   */
  onAck(ack) {
    const label = ack.command.replace(/^(calib|contact|teleop)\./, '');
    logEvent(
      commandTag(ack.command),
      ack.ok ? `${label} 완료` : `${label} 거절 — ${ack.reason ?? '사유 없음'}`,
      ack.ok ? 'info' : 'warn',
    );
  },

  onStatus(status) {
    watcher.observeLink(status.phase, status.endpoint);
    useTelemetryStore.setState({ link: status });
  },
});

adapter.start();

if (import.meta.hot) {
  import.meta.hot.dispose(() => adapter.stop());
}

export { adapter };
