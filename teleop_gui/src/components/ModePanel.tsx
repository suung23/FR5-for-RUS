import { config } from '../telemetry/config';
import type { ContactSnapshot } from '../telemetry/contactState';
import type { RobotTelemetry } from '../telemetry/types';
import {
  ROBOT_STATE_LABEL,
  SAFETY_STATE_LABEL,
  severityOfSafety,
  type Severity,
} from '../store/selectors';
import styles from './ModePanel.module.css';

interface Props {
  telemetry: RobotTelemetry;
  contact: ContactSnapshot;
  connected: boolean;
  stale: boolean;
}

/**
 * The loudest element on screen.
 *
 * Two facts are shown side by side because they answer different questions and
 * can disagree: what the robot says it is doing, and what the force sensor says
 * about contact. When they disagree that is itself information — the force path
 * is a separate USB link that does not pass through the robot controller — so
 * the panel shows both rather than reconciling them into one badge.
 */
export function ModePanel({ telemetry, contact, connected, stale }: Props) {
  const unknown = !connected || stale;
  const safetySeverity: Severity = unknown ? 'unknown' : severityOfSafety(telemetry.safetyState);
  const stage = contact.phase === 'contact' ? 'Contact' : 'Approach';
  const stageSeverity: Severity = unknown
    ? 'unknown'
    : contact.phase === 'contact'
      ? 'caution'
      : 'nominal';

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Operating mode</span>
        <span className={`dot dot--${safetySeverity}`} />
      </div>

      <div className={`panel__body ${styles.body}`}>
        <div className={styles.primary}>
          <div className={`${styles.tile} ${styles[`tile--${stageSeverity}`]}`}>
            <span className="label">Stage</span>
            <strong className={styles.tileValue}>{unknown ? '—' : stage}</strong>
            <span className={styles.tileNote}>
              {unknown
                ? 'no telemetry'
                : contact.phase === 'contact'
                  ? `above ${config.contactEnterN.toFixed(1)} N`
                  : `below ${config.contactEnterN.toFixed(1)} N`}
            </span>
            {/* The confirmation window is short but visible — an operator who
                sees the bar move understands why the stage did not flip on a
                single spike. */}
            <div
              className={styles.confirm}
              role="progressbar"
              aria-valuenow={Math.round(contact.confirmProgress * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Stage confirmation window"
            >
              <span style={{ width: `${contact.confirmProgress * 100}%` }} />
            </div>
          </div>

          <div className={`${styles.tile} ${styles[`tile--${safetySeverity}`]}`}>
            <span className="label">Safety</span>
            <strong className={styles.tileValue}>
              {unknown || !telemetry.safetyState
                ? '—'
                : SAFETY_STATE_LABEL[telemetry.safetyState]}
            </strong>
            <span className={styles.tileNote}>
              {unknown ? 'no telemetry' : 'reported by controller'}
            </span>
          </div>
        </div>

        <dl className={styles.facts}>
          <div className={styles.fact}>
            <dt className="label">Robot state</dt>
            <dd>
              {unknown || !telemetry.robotState ? '—' : ROBOT_STATE_LABEL[telemetry.robotState]}
            </dd>
          </div>
          <div className={styles.fact}>
            <dt className="label">Contact latch</dt>
            <dd className={contact.hasContacted ? styles.latched : undefined}>
              {contact.hasContacted ? 'Engaged this session' : 'Not engaged'}
            </dd>
          </div>
          <div className={styles.fact}>
            <dt className="label">Velocity scaling</dt>
            <dd>
              {contact.hasContacted ? 'Contact limits expected' : 'Freespace permitted'}
            </dd>
          </div>
        </dl>

        <p className={styles.footnote}>
          Stage is derived from the force sensor, which reaches this workstation
          on its own USB link. It is not routed through the robot controller and
          can disagree with the reported robot state.
        </p>
      </div>
    </section>
  );
}
