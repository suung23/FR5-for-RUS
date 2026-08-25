import type { ContactPhase } from '../telemetry/contactState';
import type { LinkPhase, RobotState, SafetyState } from '../telemetry/types';

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

  observeLink(phase: LinkPhase, endpoint: string): void {
    if (phase === this.link) return;
    const previous = this.link;
    this.link = phase;
    if (previous === undefined && phase !== 'connected') return;
    const severity: EventSeverity =
      phase === 'error' ? 'alarm' : phase === 'connected' ? 'info' : 'warn';
    logEvent('LINK', `${phase} — ${endpoint}`, severity);
  }

  observeStage(stage: ContactPhase, latched: boolean, normalForceN: number): void {
    if (stage !== this.stage) {
      const previous = this.stage;
      this.stage = stage;
      if (previous !== undefined) {
        logEvent(
          'STAGE',
          `${previous} to ${stage} at ${normalForceN.toFixed(2)} N`,
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
