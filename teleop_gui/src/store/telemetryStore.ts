import { create } from 'zustand';
import { RobotTelemetryAdapter } from '../telemetry/adapter';
import { config } from '../telemetry/config';
import { solveChain } from '../telemetry/fr5Model';
import { ContactDetector, type ContactSnapshot } from '../telemetry/contactState';
import type { LinkStatus, RobotTelemetry, WrenchSample } from '../telemetry/types';
import { logEvent, TransitionWatcher } from './events';

/** One point in the rolling chart buffer. */
export interface ForcePoint {
  t: number;
  fx: number;
  fy: number;
  fz: number;
  fn: number;
  /**
   * Normal force `[min, max]` over the span this point covers.
   *
   * A point drawn from a waveform bin stands for ten milliseconds of a 1 kHz
   * stream, not for one sample, so it has a width as well as a value. The plot
   * shades that width; without it the folding would silently discard exactly
   * the excursions it exists to preserve. Equal to `fn` on both sides when the
   * frame carried a single reading and there is no width to show.
   */
  fnLo: number;
  fnHi: number;
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
 */
const CHART_HZ = 25;

interface TelemetryState {
  telemetry: RobotTelemetry;
  wrench: WrenchSample | null;
  contact: ContactSnapshot;
  link: LinkStatus | null;
  history: ForcePoint[];
  /** Largest |F_n| seen since the last operator reset. */
  peakNormalForceN: number;
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
  /**
   * Operator's display zero, or null when the readings are untared.
   *
   * With no calibration the sensor sits a few tenths of a newton off zero
   * while nothing is touching the probe, and a plate labelled "contact force"
   * reading 0.2 N when there is no contact is simply wrong on its face.
   *
   * **This is a display zero and nothing more.** The control stack keeps
   * judging contact on its own untared reading, which is what keeps the robot's
   * thresholds honest — so the console says, on the plate, that it is showing a
   * zeroed number. `mode` records which pipeline stage the offset was taken
   * from; an offset taken on raw channels means nothing once compensation
   * arrives, so it is dropped rather than silently misapplied.
   */
  forceZero: { vector: [number, number, number]; mode: 'raw' | 'compensated'; at: number } | null;

  setPaused(paused: boolean): void;
  resetSession(): void;
  /** Take the current reading as zero. Refused while the contact latch holds. */
  zeroForce(): boolean;
  /** Drop the display zero and go back to what the sensor reports. */
  clearForceZero(): void;
  /** Send a console command. Returns false when there is no back channel. */
  sendCommand(command: Record<string, unknown>): boolean;
}

const detector = new ContactDetector({
  enterN: config.contactEnterN,
  releaseN: config.contactReleaseN,
  normalForceSign: config.normalForceSign,
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
  contact: detector.snapshot(),
  link: null,
  history: [],
  peakNormalForceN: 0,
  frameRateHz: 0,
  waveformHz: null,
  trajectory: [],
  paused: false,
  forceZero: null,

  setPaused: (paused) => set({ paused }),

  zeroForce: () => {
    const state = useTelemetryStore.getState();
    const sample = state.wrench;
    if (!sample) return false;
    // Zeroing while something is pressing would fold that load into the offset
    // and hide it for the rest of the session. The latch is the console's own
    // record that contact happened, so it is the right thing to ask.
    if (state.contact.hasContacted || state.contact.phase === 'contact') {
      logEvent('FORCE', 'zero refused — contact latch is engaged', 'warn');
      return false;
    }
    const compensated = sample.compensated;
    const vector: [number, number, number] = compensated
      ? [
          compensated.contactProbe[0],
          compensated.contactProbe[1],
          compensated.contactProbe[2],
        ]
      : [
          sample.force[0],
          sample.force[1],
          config.normalForceSign * sample.force[2],
        ];
    logEvent(
      'FORCE',
      `display zeroed — offset ${Math.hypot(...vector).toFixed(2)} N ` +
        `(${compensated ? 'compensated' : 'raw'} channels)`,
    );
    // The peak hold is a session record of how hard the probe pressed. Against
    // a new zero the old peak means nothing, so it goes with it.
    set({
      forceZero: { vector, mode: compensated ? 'compensated' : 'raw', at: Date.now() },
      peakNormalForceN: 0,
    });
    return true;
  },

  clearForceZero: () => {
    if (useTelemetryStore.getState().forceZero === null) return;
    logEvent('FORCE', 'display zero cleared — showing what the sensor reports');
    set({ forceZero: null, peakNormalForceN: 0 });
  },

  sendCommand: (command) => {
    const sent = adapter.sendCommand(command);
    // The tag names the subsystem the command belongs to. A teleoperation
    // change filed under CALIB would be unfindable in the log afterwards.
    const tag = String(command.command ?? '').startsWith('teleop.') ? 'TELEOP' : 'CALIB';
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
      peakNormalForceN: 0,
      forceZero: null,
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

  onWrench(sample) {
    const state = useTelemetryStore.getState();
    if (state.paused) return;

    // Contact classification always runs on the raw F_z. It is a physical
    // judgement, not a display preference, so it must not follow anything the
    // operator changed about how the chart is drawn.
    const dtMs = lastWrenchAt === null ? 0 : sample.timestamp - lastWrenchAt;
    lastWrenchAt = sample.timestamp;
    const contact = detector.update(sample.force[2], dtMs);
    watcher.observeStage(contact.phase, contact.hasContacted, contact.normalForceN);

    // The displayed reading is the mean of its window; the peak must not be.
    // At one frame a second a spike inside the window would be averaged away,
    // and the peak-hold exists to catch exactly that. The bridge sends the
    // window's own extremes alongside the mean, so the hold sees them.
    const windowPeak = sample.forceExtremes
      ? Math.max(
          Math.abs(config.normalForceSign * sample.forceExtremes[0][2]),
          Math.abs(config.normalForceSign * sample.forceExtremes[1][2]),
        )
      : Math.abs(contact.normalForceN);

    const patch: Partial<TelemetryState> = {
      wrench: sample,
      contact,
      peakNormalForceN: Math.max(state.peakNormalForceN, windowPeak),
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
        // The bin's own edges under the normal-force convention. A negative
        // sign swaps which extreme is the larger press, so they are ordered
        // after the flip rather than assumed.
        const lo = config.normalForceSign * bin.fzMin;
        const hi = config.normalForceSign * bin.fzMax;
        // Ages are relative to the frame, and the frame is stamped on arrival,
        // so network jitter could otherwise place a bin behind its predecessor.
        const t = Math.max(lastPointAt + 1, sample.timestamp - bin.ageMs);
        lastPointAt = t;
        pendingPoints.push({
          t,
          fx: bin.fx,
          fy: bin.fy,
          fz: bin.fz,
          fn: config.normalForceSign * bin.fz,
          fnLo: Math.min(lo, hi),
          fnHi: Math.max(lo, hi),
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
        fn: contact.normalForceN,
        fnLo: contact.normalForceN,
        fnHi: contact.normalForceN,
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
