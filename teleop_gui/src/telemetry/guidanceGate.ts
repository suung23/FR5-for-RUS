import type { ProbingMode, RobotState, SafetyState } from './types';

/**
 * Which guidance the operator has to acknowledge, and when.
 *
 * The console used to carry a small permanent guidance panel. Standing text in
 * a corner is read once and then stops being read — it becomes part of the
 * furniture. A gate that must be dismissed is read at the moment it matters,
 * which is the moment the situation changes.
 *
 * Kept out of the component so the trigger rules can be tested without
 * rendering anything.
 */

export type GateTopic = 'teleoperation' | 'contact_control' | 'safety_response';

export interface GateContent {
  topic: GateTopic;
  /** Conventional title — the mode the operator is entering. */
  title: string;
  items: string[];
}

export const GATE_CONTENT: Record<GateTopic, GateContent> = {
  teleoperation: {
    topic: 'teleoperation',
    title: 'Teleoperation',
    items: [
      'Confirm that the probe, cable, and patient-contact area are unobstructed.',
      'Verify live robot and force-sensor telemetry.',
      'Use slow, deliberate motion near patient contact.',
      'Monitor the normal-force limit continuously.',
      'Stop if force, robot state, or ultrasound telemetry becomes unavailable.',
    ],
  },
  contact_control: {
    topic: 'contact_control',
    title: 'Contact Control',
    items: [
      'Confirm stable probe contact before enabling force regulation.',
      'Monitor normal force and image quality continuously.',
      'Do not oppose controller motion along the probe normal axis.',
      'Use only deliberate lateral or rotational adjustment.',
      'Stop and reassess if force oscillation, discomfort, or unexpected resistance occurs.',
    ],
  },
  safety_response: {
    topic: 'safety_response',
    title: 'Safety Response',
    items: [
      'Confirm that robot motion has stopped.',
      'Assess the probe position and patient contact.',
      'Review the displayed fault or recovery message.',
      'Do not resume operation until robot and sensor telemetry are normal.',
    ],
  },
};

export interface GateObservation {
  robotState: RobotState | undefined;
  safetyState: SafetyState | undefined;
  probingMode: ProbingMode | undefined;
}

/**
 * Decides when a gate is owed.
 *
 * Holds the previous observation and reports a topic only on a transition into
 * a state that needs acknowledgement — never continuously, or the operator
 * could not work at all.
 *
 * A fault outranks a mode change: if the robot faults while entering contact
 * control, the safety checklist is the one that matters.
 */
export class GateTrigger {
  private previous: GateObservation | null = null;
  private faulted = false;

  /** Startup always owes one. */
  initialTopic(): GateTopic {
    return 'teleoperation';
  }

  /**
   * Feed the current state.
   *
   * Returns the topic to show, or `null` when nothing new is owed.
   */
  observe(now: GateObservation): GateTopic | null {
    const previous = this.previous;
    this.previous = { ...now };

    const inFault =
      now.robotState === 'fault' ||
      now.robotState === 'estop' ||
      now.safetyState === 'protective_stop' ||
      now.safetyState === 'emergency_stop';

    // Entering a fault, and again on the way out of one — the operator has to
    // acknowledge before controls come back.
    if (inFault && !this.faulted) {
      this.faulted = true;
      return 'safety_response';
    }
    if (!inFault && this.faulted) {
      this.faulted = false;
      return 'safety_response';
    }
    if (inFault) return null;

    if (!previous) return null;

    if (
      previous.probingMode !== 'contact_probing' &&
      now.probingMode === 'contact_probing'
    ) {
      return 'contact_control';
    }

    if (previous.robotState !== 'teleop' && now.robotState === 'teleop') {
      return 'teleoperation';
    }

    return null;
  }
}
