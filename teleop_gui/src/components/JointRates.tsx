import { useMemo } from 'react';
import { JOINT_LIMITS_DEG, JOINT_NAMES, jointTravelFraction, marginToLimit } from '../telemetry/fr5Model';
import { fixed, signed } from '../lib/format';
import type { RobotTelemetry } from '../telemetry/types';
import styles from './JointRates.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
}

const RAD_TO_DEG = 180 / Math.PI;
const NEAR_LIMIT_DEG = 10;

/**
 * Joint travel and rate, one row per axis.
 *
 * The travel indicator is a rule with a navy marker at the joint's position
 * within its own range — not a filled bar, because position in a signed range
 * is not a magnitude and a fill would read as one. Each axis is scaled to its
 * own limits: J2 travels 350 degrees and J6 travels 330, so a shared scale
 * would make equal margins look unequal.
 */
export function JointRates({ telemetry, available }: Props) {
  const rows = useMemo(
    () =>
      JOINT_NAMES.map((name, i) => {
        const q = available ? telemetry.jointPositions?.[i] : undefined;
        const v = available ? telemetry.jointVelocities?.[i] : undefined;
        const known = typeof q === 'number';
        return {
          name,
          index: i,
          positionDeg: known ? q * RAD_TO_DEG : undefined,
          rateDeg: typeof v === 'number' ? v * RAD_TO_DEG : undefined,
          marginDeg: known ? marginToLimit(i, q) * RAD_TO_DEG : undefined,
          fraction: known ? jointTravelFraction(i, q) : undefined,
        };
      }),
    [telemetry, available],
  );

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Joint travel</span>
        <span className="plate__aside">per-axis range from probe.yaml</span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Axis</th>
              <th scope="col">Pos</th>
              <th scope="col">Rate</th>
              <th scope="col">Margin</th>
              <th scope="col" className={styles.rangeHead}>
                Travel within limits
              </th>
              <th scope="col" className={styles.limitHead}>
                Range
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const near = row.marginDeg !== undefined && row.marginDeg < NEAR_LIMIT_DEG;
              return (
                <tr key={row.name} className={near ? styles.near : undefined}>
                  <th scope="row">{row.name}</th>
                  {/* Position and rate both cross zero, so both carry a sign
                      at all times — see lib/format. Margin cannot go negative
                      and keeps its width without one. */}
                  <td className="num">{signed(row.positionDeg, 2)}</td>
                  <td className="num">{signed(row.rateDeg, 1)}</td>
                  <td className="num">
                    {fixed(row.marginDeg, 1)}
                    {near ? <span className={styles.tag}>LIM</span> : null}
                  </td>
                  <td>
                    <div className={styles.range}>
                      <span className={styles.rangeRule} />
                      <span className={styles.rangeEnd} style={{ left: 0 }} />
                      <span className={styles.rangeEnd} style={{ left: '100%' }} />
                      {row.fraction !== undefined ? (
                        <span
                          className={`${styles.marker} ${near ? styles.markerNear : ''}`}
                          style={{ left: `${row.fraction * 100}%` }}
                        />
                      ) : null}
                    </div>
                  </td>
                  <td className={`num ${styles.limits}`}>
                    {JOINT_LIMITS_DEG.lower[row.index]} … {JOINT_LIMITS_DEG.upper[row.index]}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
