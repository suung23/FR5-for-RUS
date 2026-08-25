import { useMemo } from 'react';
import { config } from '../telemetry/config';
import type { ForcePoint } from '../store/telemetryStore';
import type { ContactSnapshot } from '../telemetry/contactState';
import styles from './ModeTimeline.module.css';

interface Props {
  history: ForcePoint[];
  contact: ContactSnapshot;
}

interface Band {
  phase: 'approach' | 'contact';
  fromPct: number;
  widthPct: number;
  seconds: number;
}

/**
 * Mode-transition timeline.
 *
 * Replays the stage classifier over the retained force history and draws the
 * result as banded track: approach is open, contact is filled black. It answers
 * a question the trend chart does not — *when* did the stage change, and how
 * long has it held — without asking the operator to read a threshold crossing
 * off a curve.
 *
 * This is a reconstruction from the buffered samples, not the live detector's
 * own record, so it is labelled as such: the decimated buffer can miss a
 * crossing shorter than its sample period.
 */
export function ModeTimeline({ history, contact }: Props) {
  const { bands, spanSeconds } = useMemo(() => buildBands(history), [history]);

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Mode transitions</span>
        <span className="plate__aside">reconstructed from 20 s buffer</span>
      </div>

      <div className={`plate__body ${styles.body}`}>
        <div className={styles.trackRow}>
          <span className={styles.trackLabel}>STAGE</span>
          <div className={styles.track} role="img" aria-label="Stage over the last 20 seconds">
            {bands.length === 0 ? (
              <span className={styles.trackEmpty}>NO DATA</span>
            ) : (
              bands.map((band, i) => (
                <span
                  key={i}
                  className={band.phase === 'contact' ? styles.bandContact : styles.bandApproach}
                  style={{ left: `${band.fromPct}%`, width: `${band.widthPct}%` }}
                  title={`${band.phase} · ${band.seconds.toFixed(1)} s`}
                >
                  {band.widthPct > 12 ? (
                    <span className={styles.bandText}>
                      {band.phase === 'contact' ? 'CONTACT' : 'APPROACH'}
                    </span>
                  ) : null}
                </span>
              ))
            )}
          </div>
        </div>

        <div className={styles.axis}>
          <span>-{spanSeconds.toFixed(0)}s</span>
          <span className={styles.axisMid}>-{(spanSeconds / 2).toFixed(0)}s</span>
          <span>NOW</span>
        </div>

        <dl className={styles.readout}>
          <Row label="Current stage" value={contact.phase === 'contact' ? 'CONTACT' : 'APPROACH'} />
          <Row
            label="Transitions in buffer"
            value={String(Math.max(0, bands.length - 1))}
          />
          <Row
            label="Threshold"
            value={`enter ${config.contactEnterN.toFixed(1)} N · release ${config.contactReleaseN.toFixed(1)} N`}
          />
          <Row
            label="Latch"
            value={contact.hasContacted ? 'ENGAGED' : 'NOT ENGAGED'}
            strong={contact.hasContacted}
          />
        </dl>

        <div className={styles.bands}>
          <div className={styles.bandsHead}>BANDS IN BUFFER · NEWEST FIRST</div>
          {bands.length === 0 ? (
            <p className={styles.bandEmpty}>NO DATA</p>
          ) : (
            <table className={styles.bandTable}>
              <tbody>
                {[...bands].reverse().map((band, i) => (
                  <tr
                    key={`${band.fromPct}-${i}`}
                    className={band.phase === 'contact' ? styles.bandContactRow : undefined}
                  >
                    <th scope="row">{band.phase === 'contact' ? 'CONTACT' : 'APPROACH'}</th>
                    <td className="num">{band.seconds.toFixed(1)} s</td>
                    <td className="num">
                      {(-(spanSeconds * (1 - band.fromPct / 100))).toFixed(1)} s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={`field ${strong ? 'field--warn' : ''}`}>
      <dt className="field__label">{label}</dt>
      <dd className={`field__value num ${styles.rowValue}`}>{value}</dd>
    </div>
  );
}

/**
 * Segment the buffer into stage bands.
 *
 * Uses the same enter/release thresholds as the live detector but without its
 * confirmation windows, which cannot be reproduced from a decimated buffer.
 * The consequence is that a very brief crossing may appear here as a hard edge
 * where the detector would have rejected it.
 */
function buildBands(history: ForcePoint[]): { bands: Band[]; spanSeconds: number } {
  if (history.length < 2) return { bands: [], spanSeconds: 20 };

  const t0 = history[0].t;
  const t1 = history[history.length - 1].t;
  const span = Math.max(1, t1 - t0);

  const bands: Band[] = [];
  let phase: 'approach' | 'contact' = 'approach';
  let startedAt = t0;

  for (const point of history) {
    const fn = Math.abs(point.fn);
    const next: 'approach' | 'contact' =
      phase === 'approach'
        ? fn >= config.contactEnterN
          ? 'contact'
          : 'approach'
        : fn <= config.contactReleaseN
          ? 'approach'
          : 'contact';
    if (next !== phase) {
      bands.push(makeBand(phase, startedAt, point.t, t0, span));
      phase = next;
      startedAt = point.t;
    }
  }
  bands.push(makeBand(phase, startedAt, t1, t0, span));

  return { bands, spanSeconds: span / 1000 };
}

function makeBand(
  phase: 'approach' | 'contact',
  from: number,
  to: number,
  t0: number,
  span: number,
): Band {
  return {
    phase,
    fromPct: ((from - t0) / span) * 100,
    widthPct: Math.max(0.4, ((to - from) / span) * 100),
    seconds: (to - from) / 1000,
  };
}
