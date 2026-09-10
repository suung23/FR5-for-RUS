import type { RobotTelemetry } from '../telemetry/types';
import styles from './QualityScores.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
}

/**
 * The two image-quality readings, out of 100.
 *
 * They are **not** two views of one number. `Q_seg` is segmentation-based and
 * answers *is the bladder properly shown*; `Q_raw` is classical image
 * processing with no segmentation and answers *is the probe properly coupled*.
 * The force search maximises `Q_raw`; the success criterion is built from the
 * segmentation. Printing a single "quality" would hide the case the operator
 * most needs to see.
 *
 * That case is disagreement. High coupling with a low segmentation score means
 * the probe is on the tissue but pointed at nothing — move. The reverse means
 * the network found something the contact does not support — press differently
 * before believing it.
 *
 * A missing reading prints as a dash, never as zero. `Q_raw` returns *not
 * measured* when too few A-lines had ROI support, and `Q_seg` is blind whenever
 * the network produces no mask; both are "we cannot say", and an operator who
 * reads a zero there will chase a degradation that was never observed.
 */
const BANDS: { min: number; label: string }[] = [
  { min: 80, label: 'good' },
  { min: 60, label: 'usable' },
  { min: 0, label: 'poor' },
];

function band(score: number | undefined): string {
  if (score === undefined) return 'not measured';
  return BANDS.find((b) => score >= b.min)?.label ?? 'poor';
}

function Score({ label, note, value }: { label: string; note: string; value?: number }) {
  const shown = value === undefined ? null : Math.round(value * 100);
  return (
    <div className={styles.score}>
      <span className={styles.scoreLabel}>{label}</span>
      <span className={`${styles.scoreValue} ${shown === null ? styles.absent : ''}`}>
        {shown === null ? '—' : shown}
      </span>
      <span className={styles.scoreNote}>{note}</span>
      <span className={styles.scoreBand}>{band(value)}</span>
    </div>
  );
}

export function QualityScores({ telemetry, available }: Props) {
  const seg = available ? telemetry.qualitySeg : undefined;
  const raw = available ? telemetry.qualityRaw : undefined;
  // Disagreement is worth a sentence only when both were actually measured.
  const gap = seg !== undefined && raw !== undefined ? Math.round((raw - seg) * 100) : null;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Image quality</span>
        <span className="plate__aside">out of 100 · two readings</span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.scores}>
          <Score label="Q_seg" note="bladder shown" value={seg} />
          <Score label="Q_raw" note="probe coupled" value={raw} />
        </div>
        <p className={styles.state} role="status">
          {gap === null ? (
            <>
              A dash is <strong>not measured</strong>, not zero — the segmentation has no mask,
              or too few A-lines had ROI support.
            </>
          ) : gap >= 20 ? (
            <>
              Coupled but not showing the bladder ({gap} apart). This is the case the policy is
              for: <strong>move</strong>, do not press harder.
            </>
          ) : gap <= -20 ? (
            <>
              The mask is ahead of the coupling ({-gap} apart). Press differently before trusting
              it — the force search is working on <code>Q_raw</code>.
            </>
          ) : (
            <>
              The two readings agree. <code>Q_raw</code> drives the force setpoint;{' '}
              <code>Q_seg</code> is what the success criterion is built from.
            </>
          )}
        </p>
      </div>
    </section>
  );
}
