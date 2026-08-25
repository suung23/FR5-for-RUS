import { config } from './config';
import type { ContactSnapshot } from './contactState';
import type { RobotTelemetry } from './types';

/**
 * Operator guidance.
 *
 * Short, imperative, and tied to something actually observed. The rule this
 * follows: never tell the physician something the screen already shows. A line
 * earns its place only if it names what to *do*, or states a limitation of the
 * reading that is not visible in the number itself.
 *
 * Ordering is by urgency, and only the first few are shown.
 */

export type GuidanceTone = 'critical' | 'caution' | 'info';

export interface GuidanceItem {
  id: string;
  tone: GuidanceTone;
  text: string;
}

export interface GuidanceInput {
  telemetry: RobotTelemetry;
  contact: ContactSnapshot;
  connected: boolean;
  stale: boolean;
  simulated: boolean;
}

const TONE_RANK: Record<GuidanceTone, number> = {
  critical: 0,
  caution: 1,
  info: 2,
};

export function buildGuidance(input: GuidanceInput): GuidanceItem[] {
  const { telemetry, contact, connected, stale, simulated } = input;
  const items: GuidanceItem[] = [];
  const fn = Math.abs(contact.normalForceN);

  if (simulated) {
    items.push({
      id: 'simulated',
      tone: 'caution',
      text: 'Simulated telemetry. No robot or force sensor is connected — readings on this screen are generated, not measured.',
    });
  }

  if (!connected) {
    items.push({
      id: 'link-down',
      tone: 'critical',
      text: 'Telemetry link is down. Values shown are the last received and may not reflect the robot. Do not rely on this display.',
    });
  } else if (stale) {
    items.push({
      id: 'stale',
      tone: 'critical',
      text: 'No recent frames. The displayed pose is frozen — confirm the robot state at the pendant before continuing.',
    });
  }

  if (telemetry.safetyState === 'emergency_stop') {
    items.push({
      id: 'estop',
      tone: 'critical',
      text: 'Emergency stop is active. Clear the cause, then release and re-initialise at the pendant before resuming.',
    });
  } else if (telemetry.safetyState === 'protective_stop') {
    items.push({
      id: 'protective',
      tone: 'critical',
      text: 'Protective stop. Reduce the applied load and retract along the probe axis before attempting to resume.',
    });
  }

  if (connected && fn >= config.maxForceN) {
    items.push({
      id: 'force-limit',
      tone: 'critical',
      text: `Normal force is at or above the ${config.maxForceN.toFixed(1)} N limit. Withdraw along the probe axis now.`,
    });
  } else if (connected && fn >= config.warnForceN) {
    items.push({
      id: 'force-warn',
      tone: 'caution',
      text: `Normal force above ${config.warnForceN.toFixed(1)} N. Hold position; do not advance further along the probe axis.`,
    });
  }

  if (contact.phase === 'contact') {
    items.push({
      id: 'contact-scale',
      tone: 'info',
      text: 'In contact. Contact-stage velocity limits apply — relaunch teleoperation with freespace disabled if it is still running at approach speed.',
    });
  } else if (contact.hasContacted) {
    items.push({
      id: 'latched',
      tone: 'info',
      text: 'Contact has occurred this session. Lifting the probe does not return the session to the approach stage.',
    });
  }

  if (connected && contact.phase === 'approach' && !contact.hasContacted) {
    items.push({
      id: 'approach',
      tone: 'info',
      text: `Approach stage. Contact is declared at ${config.contactEnterN.toFixed(1)} N normal force.`,
    });
  }

  // Stated once, always: the force reading is not yet trustworthy in absolute
  // terms, and the operator should know why before trusting a threshold.
  items.push({
    id: 'gravity',
    tone: 'info',
    text: 'Probe weight is not compensated. The normal force includes a pose-dependent gravity term of roughly 1.5 N.',
  });

  return items.sort((a, b) => TONE_RANK[a.tone] - TONE_RANK[b.tone]);
}
