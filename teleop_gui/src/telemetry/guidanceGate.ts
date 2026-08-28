/**
 * The operator briefing, shown once per session.
 *
 * The console used to carry a small permanent guidance panel. Standing text in
 * a corner is read once and then stops being read — it becomes part of the
 * furniture. A gate that must be dismissed is read.
 *
 * It is deliberately **one-shot**: shown at startup, acknowledged, and never
 * again. An overlay that reappears mid-procedure is worse than useless — it
 * covers the live state at exactly the moment the operator is reacting to it,
 * and after the second or third time it gets dismissed unread, which is the
 * failure mode the gate existed to avoid.
 *
 * Because it is the only time guidance is shown, it carries all three
 * checklists rather than just the one for the starting mode.
 */

export interface BriefingSection {
  title: string;
  items: string[];
}

export const BRIEFING_SECTIONS: BriefingSection[] = [
  {
    title: 'Teleoperation',
    items: [
      'Confirm that the probe, cable, and contact area are unobstructed.',
      'Verify live robot and force-sensor telemetry.',
      'Use slow, deliberate motion near contact.',
      'Monitor the normal-force limit continuously.',
      'Stop if force, robot state, or ultrasound telemetry becomes unavailable.',
    ],
  },
  {
    title: 'Contact control',
    items: [
      'Force regulation engages on its own at the contact threshold.',
      'Do not oppose controller motion along the probe normal axis.',
      'Monitor normal force and image quality continuously.',
      'Stop and reassess on force oscillation or unexpected resistance.',
    ],
  },
  {
    title: 'Safety response',
    items: [
      'Confirm that robot motion has stopped.',
      'Assess the probe position and contact state.',
      'Review the displayed fault or recovery message.',
      'Do not resume until robot and sensor telemetry are normal.',
    ],
  },
];
