import { config } from '../telemetry/config';
import type { RobotState, SafetyState } from '../telemetry/types';
import type { ForcePoint } from './telemetryStore';

/**
 * Derived reads used by more than one panel.
 *
 * Kept out of the components so that "what counts as stale" or "what the mode
 * banner says" has one definition rather than one per panel.
 */

export type Severity = 'nominal' | 'caution' | 'critical' | 'unknown';

export const STALE_AFTER_MS = 1500;

export function isStale(timestamp: number, now: number): boolean {
  return timestamp === 0 || now - timestamp > STALE_AFTER_MS;
}

export function severityOfSafety(state: SafetyState | undefined): Severity {
  switch (state) {
    case 'normal':
      return 'nominal';
    case 'warning':
      return 'caution';
    case 'protective_stop':
    case 'emergency_stop':
      return 'critical';
    default:
      return 'unknown';
  }
}

export function severityOfForce(normalForceN: number): Severity {
  const magnitude = Math.abs(normalForceN);
  if (magnitude >= config.maxForceN) return 'critical';
  if (magnitude >= config.warnForceN) return 'caution';
  return 'nominal';
}

export const ROBOT_STATE_LABEL: Record<RobotState, string> = {
  idle: 'Idle',
  teleop: 'Teleoperation',
  contact: 'Contact',
  fault: 'Fault',
  estop: 'Emergency stop',
};

export const SAFETY_STATE_LABEL: Record<SafetyState, string> = {
  normal: 'Normal',
  warning: 'Warning',
  protective_stop: 'Protective stop',
  emergency_stop: 'Emergency stop',
};

/** Newest point in the buffer, or null while it is still filling. */
export function latestPoint(history: ForcePoint[]): ForcePoint | null {
  return history.length > 0 ? history[history.length - 1] : null;
}
