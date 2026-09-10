import type { RobotTelemetry } from '../telemetry/types';
import styles from './ProbingMode.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

/**
 * Contact-probing permission — and the handover to the policy.
 *
 * The two buttons send `teleop.contact_probing`, which is a **permission**, not
 * a motion command: armed, the robot may take the beam axis when it judges
 * contact; disarmed, that permission is withdrawn and a probing arm returns to
 * approach. Nothing here drives the arm. The control stack still owns the
 * velocity clamps and the watchdogs.
 *
 * What the panel prints is the mode that came back on `probing_mode`, never the
 * button that was pressed — the same rule the operator station and the in-plane
 * toggle follow. A button that lit on its own click would show an intent the
 * stack may have refused.
 *
 * **This is where the policy takes over.** `scripts/run_policy.py --start-on
 * probing` watches the same topic and begins inferring when the mode enters
 * `contact_probing*`. Entry is the handover point, so the operator must let go
 * of the stylus there: `desired_twist` has to have a single publisher, and a
 * teleop still commanding would fight the policy for the same axes.
 */
const MODE_LABEL: Record<string, string> = {
  approach: 'Approach',
  contact_probing: 'Contact probing',
  contact_probing_inplane: 'Contact probing · in-plane',
};

export function ProbingMode({ telemetry, available, onCommand }: Props) {
  const mode = telemetry.probingMode;
  // No mode means us_diff_ik has not declared one — either it is not up or this
  // bridge predates the topic. Printing "approach" would be a guess, and a wrong
  // guess here says the beam axis is free when it may not be.
  const known = mode !== undefined;
  const probing = mode === 'contact_probing' || mode === 'contact_probing_inplane';

  const send = (enabled: boolean) =>
    onCommand({ command: 'teleop.contact_probing', enabled });

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Probing mode</span>
        <span className="plate__aside">
          {known ? MODE_LABEL[mode] ?? mode : 'not declared by the stack'}
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.choices} role="group" aria-label="Contact probing permission">
          <button
            type="button"
            aria-pressed={probing}
            disabled={!known || !available}
            className={`${styles.choice} ${probing ? styles.applied : ''}`}
            onClick={() => send(true)}
          >
            <span className={styles.choiceLabel}>Arm probing</span>
            <span className={styles.choiceNote}>robot may take the beam axis on contact</span>
          </button>
          <button
            type="button"
            aria-pressed={known && !probing}
            disabled={!known || !available}
            className={`${styles.choice} ${known && !probing ? styles.applied : ''}`}
            onClick={() => send(false)}
          >
            <span className={styles.choiceLabel}>Disarm</span>
            <span className={styles.choiceNote}>withdraw permission · return to approach</span>
          </button>
        </div>

        <p className={styles.state} role="status">
          {!known ? (
            <>
              The control stack has not declared a mode. Nothing is sent until it does —
              this panel will not guess whether the beam axis is free.
            </>
          ) : probing ? (
            <>
              <span className="tag tag--strong">HANDOVER</span> the robot holds the beam axis.
              If the policy runner is up it is inferring now — <strong>let go of the
              stylus</strong>, or two publishers will fight for the same axes.
            </>
          ) : (
            <>
              Approach. The permission is withdrawn: contact will not hand the beam axis to
              the robot, and the policy stays idle.
            </>
          )}
        </p>
      </div>
    </section>
  );
}
