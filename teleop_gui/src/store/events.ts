import type { ContactPhase } from '../telemetry/contactState';
import type { LinkPhase, ProbingMode, RobotState, SafetyState } from '../telemetry/types';

/**
 * Event log.
 *
 * Records transitions, never levels. A line is written when something changes
 * — the link came up, the stage crossed, the controller declared a stop — so
 * the log stays short enough to be read at a glance instead of scrolling past
 * a thousand identical samples.
 *
 * Severity is carried as a word, not a colour, because the console renders it
 * on a monochrome plate.
 */

export type EventSeverity = 'info' | 'warn' | 'alarm';

export interface LogEvent {
  id: number;
  at: number;
  severity: EventSeverity;
  /** Short uppercase source tag, e.g. LINK, STAGE, SAFETY. */
  source: string;
  text: string;
}

const MAX_EVENTS = 200;

let nextId = 1;
let events: LogEvent[] = [];
const listeners = new Set<(events: LogEvent[]) => void>();

export function subscribeEvents(fn: (events: LogEvent[]) => void): () => void {
  listeners.add(fn);
  fn(events);
  return () => listeners.delete(fn);
}

export function getEvents(): LogEvent[] {
  return events;
}

export function logEvent(
  source: string,
  text: string,
  severity: EventSeverity = 'info',
): void {
  const event: LogEvent = { id: nextId++, at: Date.now(), severity, source, text };
  // Newest first: the bottom strip shows only a few rows, and the useful ones
  // are always the most recent.
  events = [event, ...events].slice(0, MAX_EVENTS);
  listeners.forEach((fn) => fn(events));
}

export function clearEvents(): void {
  events = [];
  nextId = 1;
  listeners.forEach((fn) => fn(events));
}

/**
 * Transition tracker.
 *
 * Holds the previous value of each watched field and emits a line only when it
 * actually changes. Kept here rather than in the store so the comparison rules
 * live next to the log they feed.
 */
export class TransitionWatcher {
  private link?: LinkPhase;
  private stage?: ContactPhase;
  private latched = false;
  private safety?: SafetyState;
  private robot?: RobotState;
  private probing?: ProbingMode;

  observeLink(phase: LinkPhase, endpoint: string): void {
    if (phase === this.link) return;
    const previous = this.link;
    this.link = phase;
    if (previous === undefined && phase !== 'connected') return;
    const severity: EventSeverity =
      phase === 'error' ? 'alarm' : phase === 'connected' ? 'info' : 'warn';
    logEvent('LINK', `${phase} — ${endpoint}`, severity);
  }

  observeStage(stage: ContactPhase, latched: boolean, contactForceN: number): void {
    if (stage !== this.stage) {
      const previous = this.stage;
      this.stage = stage;
      if (previous !== undefined) {
        logEvent(
          'STAGE',
          `${previous} to ${stage} at ${contactForceN.toFixed(2)} N`,
          stage === 'contact' ? 'warn' : 'info',
        );
      }
    }
    if (latched && !this.latched) {
      this.latched = true;
      logEvent('STAGE', 'contact latch engaged — freespace scaling withdrawn', 'warn');
    }
    if (!latched && this.latched) {
      this.latched = false;
      logEvent('STAGE', 'contact latch cleared by operator', 'info');
    }
  }

  /**
   * The velocity-limit mode the control stack declares.
   *
   * Distinct from `observeStage`, which watches the console's own contact
   * classification. This one is the robot speaking: it is what the arm is
   * actually clamping to. The two can disagree, and when they do the log is
   * where that shows — which is the whole reason both are recorded rather than
   * one being derived from the other.
   *
   * Entering is a `warn`: nothing is wrong, but the machine the operator is
   * driving has changed under them. Leaving is `info` — it is a return to the
   * state the session started in.
   */
  observeProbingMode(mode: ProbingMode | undefined): void {
    if (!mode || mode === this.probing) return;
    const previous = this.probing;
    this.probing = mode;
    // The first frame tells us where the robot already was; that is not a
    // transition and must not be logged as one. A console attached mid-session
    // would otherwise announce a switch that happened before it was watching.
    if (previous === undefined) return;
    if (mode === 'contact_probing') {
      logEvent(
        'MODE',
        'approach to contact probing — velocity limits tightened, ' +
          'the robot is holding force on the penetration axis',
        'warn',
      );
    } else {
      logEvent('MODE', 'contact probing to approach — force hold released', 'info');
    }
  }

  observeSafety(state: SafetyState | undefined): void {
    if (!state || state === this.safety) return;
    this.safety = state;
    const severity: EventSeverity =
      state === 'emergency_stop' || state === 'protective_stop'
        ? 'alarm'
        : state === 'warning'
          ? 'warn'
          : 'info';
    logEvent('SAFETY', state.replace(/_/g, ' '), severity);
  }

  observeRobot(state: RobotState | undefined): void {
    if (!state || state === this.robot) return;
    this.robot = state;
    logEvent('ROBOT', state, state === 'fault' || state === 'estop' ? 'alarm' : 'info');
  }

  reset(): void {
    this.latched = false;
  }
}
