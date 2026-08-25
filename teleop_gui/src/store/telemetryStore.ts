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
}

/** Seconds of force history retained for the plots. */
const HISTORY_SECONDS = 20;
/** Seconds of flange path drawn in the workspace. */
const TRAJECTORY_SECONDS = 10;
/** Path sample rate. Denser than this and the polyline is noise, not a path. */
const TRAJECTORY_HZ = 10;
/** Chart sample rate. Above this the lines are denser than the pixels. */
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
  /** Recent flange positions in metres, oldest first, for the workspace path. */
  trajectory: [number, number, number][];
  paused: boolean;

  setPaused(paused: boolean): void;
  resetSession(): void;
}

const detector = new ContactDetector({
  enterN: config.contactEnterN,
  releaseN: config.contactReleaseN,
  normalForceSign: config.normalForceSign,
});

const watcher = new TransitionWatcher();
let lastWrenchAt: number | null = null;
let lastChartAt = 0;
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
  trajectory: [],
  paused: false,

  setPaused: (paused) => set({ paused }),

  resetSession: () => {
    detector.reset(true);
    watcher.reset();
    logEvent('SESSION', 'operator reset — history, peak hold and contact latch cleared');
    set({
      history: [],
      trajectory: [],
      peakNormalForceN: 0,
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

    const patch: Partial<TelemetryState> = {
      wrench: sample,
      contact,
      peakNormalForceN: Math.max(state.peakNormalForceN, Math.abs(contact.normalForceN)),
    };

    // Decimate into the chart buffer. The sensor streams far faster than any
    // plot needs, and pushing every sample would spend the frame budget on
    // points that land on the same pixel column.
    if (sample.timestamp - lastChartAt >= 1000 / CHART_HZ) {
      lastChartAt = sample.timestamp;
      const cutoff = sample.timestamp - HISTORY_SECONDS * 1000;
      const next = state.history.filter((p) => p.t >= cutoff);
      next.push({
        t: sample.timestamp,
        fx: sample.force[0],
        fy: sample.force[1],
        fz: sample.force[2],
        fn: contact.normalForceN,
      });
      patch.history = next;
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
