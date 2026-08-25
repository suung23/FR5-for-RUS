import { create } from 'zustand';
import { RobotTelemetryAdapter } from '../telemetry/adapter';
import { config } from '../telemetry/config';
import { ContactDetector, type ContactSnapshot } from '../telemetry/contactState';
import type { LinkStatus, RobotTelemetry, WrenchSample } from '../telemetry/types';

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
  paused: boolean;

  setPaused(paused: boolean): void;
  resetSession(): void;
}

const detector = new ContactDetector({
  enterN: config.contactEnterN,
  releaseN: config.contactReleaseN,
  normalForceSign: config.normalForceSign,
});

let lastWrenchAt: number | null = null;
let lastChartAt = 0;
const frameStamps: number[] = [];

export const useTelemetryStore = create<TelemetryState>((set) => ({
  telemetry: { timestamp: 0, connected: false },
  wrench: null,
  contact: detector.snapshot(),
  link: null,
  history: [],
  peakNormalForceN: 0,
  frameRateHz: 0,
  paused: false,

  setPaused: (paused) => set({ paused }),

  resetSession: () => {
    detector.reset(true);
    set({
      history: [],
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
    useTelemetryStore.setState({
      telemetry: frame,
      frameRateHz: frameStamps.length,
    });
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
    useTelemetryStore.setState({ link: status });
  },
});

adapter.start();

if (import.meta.hot) {
  import.meta.hot.dispose(() => adapter.stop());
}

export { adapter };
