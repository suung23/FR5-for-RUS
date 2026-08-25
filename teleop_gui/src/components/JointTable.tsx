import {
  JOINT_LIMITS_DEG,
  JOINT_NAMES,
  jointTravelFraction,
  marginToLimit,
} from '../telemetry/kinematics';
import styles from './JointTable.module.css';

interface Props {
  jointPositions?: number[];
  jointVelocities?: number[];
  available: boolean;
}

const RAD_TO_DEG = 180 / Math.PI;
const NEAR_LIMIT_DEG = 10;

/**
 * Per-joint readout with travel shown in place.
 *
 * The bar behind each row is position within that joint's own range, not a
 * shared scale — J2 travels 350 degrees and J6 travels 330, so a common axis
 * would make identical margins look different. What the operator needs is
 * "how close is this joint to its own stop", and that is what the bar shows.
 */
export function JointTable({ jointPositions, jointVelocities, available }: Props) {
  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Joints · 6-DOF</span>
        <span className={styles.units}>deg · deg/s</span>
      </div>
      <div className={`panel__body panel__body--flush ${styles.body}`}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Axis</th>
              <th scope="col">Position</th>
              <th scope="col">Velocity</th>
              <th scope="col">Margin</th>
              <th scope="col" className={styles.rangeHead}>
                Travel
              </th>
            </tr>
          </thead>
          <tbody>
            {JOINT_NAMES.map((name, i) => {
              const q = available ? jointPositions?.[i] : undefined;
              const v = available ? jointVelocities?.[i] : undefined;
              const known = typeof q === 'number';
              const marginDeg = known ? marginToLimit(i, q) * RAD_TO_DEG : undefined;
              const nearLimit = marginDeg !== undefined && marginDeg < NEAR_LIMIT_DEG;
              const fraction = known ? jointTravelFraction(i, q) : 0.5;

              return (
                <tr key={name} className={nearLimit ? styles.warnRow : undefined}>
                  <th scope="row" className={styles.axis}>
                    {name}
                  </th>
                  <td className={`num ${styles.value}`}>
                    {known ? (q * RAD_TO_DEG).toFixed(2) : '—'}
                  </td>
                  <td className={`num ${styles.velocity}`}>
                    {typeof v === 'number' ? (v * RAD_TO_DEG).toFixed(1) : '—'}
                  </td>
                  <td className={`num ${styles.margin} ${nearLimit ? styles.marginWarn : ''}`}>
                    {marginDeg !== undefined ? marginDeg.toFixed(1) : '—'}
                  </td>
                  <td className={styles.rangeCell}>
                    <div className={styles.range} title={`${JOINT_LIMITS_DEG.lower[i]}° … ${JOINT_LIMITS_DEG.upper[i]}°`}>
                      <span className={styles.rangeTrack} />
                      {known ? (
                        <span
                          className={`${styles.rangeDot} ${nearLimit ? styles.rangeDotWarn : ''}`}
                          style={{ left: `${fraction * 100}%` }}
                        />
                      ) : null}
                    </div>
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
