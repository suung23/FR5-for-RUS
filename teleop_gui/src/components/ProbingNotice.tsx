import { useEffect, useRef, useState } from 'react';
import { config } from '../telemetry/config';
import styles from './ProbingNotice.module.css';

/**
 * Standing notice for contact probing.
 *
 * Contact probing changes what the operator's hands do, not just what the
 * numbers say: the velocity limits tighten by fifteen times, the robot closes
 * a force loop on the penetration axis by itself, and the other five axes are
 * commanded to zero. The right-hand column has always carried a `Velocity
 * limits` field for this, but a field is read when it is looked at, and the
 * moment this changes is the moment the operator is watching the arm.
 *
 * **Not an overlay.** The briefing gate is the one modal in this console and it
 * is deliberately once-only: an overlay that appears mid-procedure covers the
 * live state at the moment the operator is reacting to it, and gets dismissed
 * unread. This is a bar in the same slot as the HOLD bar — it displaces the
 * view rather than covering it, and it stays for as long as the mode does.
 *
 * **Not dismissable.** There is nothing to acknowledge. It is a statement of
 * what the machine is doing, and it goes away when the machine stops doing it.
 *
 * The elapsed time is there because force is being applied continuously and
 * nobody is commanding it. How long that has been true is a thing an operator
 * should not have to estimate.
 */
export function ProbingNotice() {
  const since = useSince();

  return (
    // `role="status"` and not `alert`: this is a change of régime, not a fault.
    // An assertive live region would interrupt a screen reader mid-sentence for
    // something that is not an emergency.
    <div className={styles.notice} role="status">
      <span className={`tag tag--strong ${styles.tag}`}>CONTACT PROBING</span>
      <div className={styles.body}>
        <p className={styles.line}>
          The robot is holding <b>‖F‖ {config.targetForceN.toFixed(1)} ±{' '}
          {config.targetBandN.toFixed(1)} N</b> on the penetration axis by itself. Your
          other five axes are commanded to zero — stylus motion will not move the arm
          while this is shown.
        </p>
        <p className={styles.line}>
          Release the deadman to retreat. Approach limits return on their own once ‖F‖
          stays under {config.contactProbingReleaseN.toFixed(1)} N for{' '}
          {config.contactProbingReleaseS.toFixed(1)} s.
        </p>
      </div>
      <span className={styles.elapsed}>
        <span className={styles.elapsedLabel}>HELD</span>
        <span className="num">{since}</span>
      </span>
    </div>
  );
}

/**
 * Time since this notice mounted, as `m:ss`.
 *
 * Mount time is the entry into contact probing, because the notice is rendered
 * only while the mode holds and React unmounts it on the way out. Measuring it
 * here rather than from a timestamp in the store keeps the clock and the thing
 * it is timing from ever disagreeing about when they started.
 */
function useSince(): string {
  const start = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setElapsed(Date.now() - start.current), 500);
    return () => clearInterval(id);
  }, []);

  const total = Math.floor(elapsed / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}
