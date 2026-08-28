import type { RobotTelemetry } from '../telemetry/types';
import styles from './OperatorFrame.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

/**
 * Where the operator is standing, relative to the robot.
 *
 * The hand mapping has always assumed the operator faces the same way the
 * robot does. Standing opposite it breaks that assumption: the operator's
 * forward is the robot's back and the operator's right is its left, which from
 * the seat reads as "I push and it comes at me".
 *
 * The fix is one rotation about the base vertical, applied to translation and
 * rotation alike. 180 degrees is what "mirror" means here — **not** a
 * reflection. A reflection flips left/right and leaves fore/aft alone; walking
 * around to the other side flips both, and that is a rotation. A reflection
 * would also break rotation commands, which are pseudovectors, so the control
 * stack does not offer one.
 *
 * The penetration axis is untouched at every setting. Only the two horizontal
 * axes turn.
 *
 * **Nothing here moves the robot.** It changes which way a hand motion the
 * operator is already making gets read, and even that waits for the next grip:
 * flipping axes mid-motion would send the arm the opposite way while the
 * operator's reflex correction makes it worse.
 */
const STATIONS: { yaw: number; label: string; note: string }[] = [
  { yaw: 0, label: 'Alongside', note: 'same heading as the robot' },
  { yaw: 90, label: 'Left side', note: 'quarter turn' },
  { yaw: 180, label: 'Facing · mirror', note: 'fore/aft and left/right flip' },
  { yaw: -90, label: 'Right side', note: 'quarter turn' },
];

export function OperatorFrame({ telemetry, available, onCommand }: Props) {
  const frame = telemetry.teleopFrame;
  const applied = frame?.operatorYawDeg;
  const pending = frame?.pendingYawDeg ?? null;
  // No frame means the IK node has not declared one — either it is not up or
  // this bridge predates the topic. Showing "0" would be a guess, and a wrong
  // guess here points the arm the other way.
  const known = frame !== undefined;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Operator station</span>
        <span className="plate__aside">
          {known ? `rotation about base vertical · ${applied}°` : 'not declared by the stack'}
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.choices} role="group" aria-label="Operator station">
          {STATIONS.map((station) => {
            const isApplied = known && applied === station.yaw;
            const isPending = pending === station.yaw;
            return (
              <button
                key={station.yaw}
                type="button"
                aria-pressed={isApplied}
                disabled={!known || !available}
                className={`${styles.choice} ${isApplied ? styles.applied : ''} ${
                  isPending ? styles.pending : ''
                }`}
                onClick={() =>
                  onCommand({ command: 'teleop.operator_frame', yawDeg: station.yaw })
                }
              >
                <span className={styles.choiceLabel}>{station.label}</span>
                <span className={styles.choiceYaw}>{station.yaw}°</span>
                <span className={styles.choiceNote}>{station.note}</span>
              </button>
            );
          })}
        </div>

        <p className={styles.state} role="status">
          {!known ? (
            <>
              The control stack has not declared a mapping. Nothing is sent until it
              does — this panel will not guess which way &ldquo;right&rdquo; is.
            </>
          ) : pending !== null ? (
            <>
              <span className="tag tag--strong">PENDING</span> {pending}° takes effect at the{' '}
              <strong>next grip</strong>. Release the deadman and take it again. Until then the
              arm still reads your hand at {applied}°.
            </>
          ) : frame?.mirrored ? (
            <>
              <span className="tag tag--green">MIRROR</span> Facing the robot. Fore/aft and
              left/right are flipped; the penetration axis is not.
            </>
          ) : (
            <>
              <span className="tag tag--off">DIRECT</span> Alongside the robot — hand axes map
              straight through.
            </>
          )}
        </p>

        <dl className={styles.detail}>
          <div>
            <dt>Translation</dt>
            <dd>{frame?.linearFrame ?? '—'}</dd>
          </div>
          <div>
            <dt>Rotation</dt>
            <dd>{frame?.angularFrame ?? '—'}</dd>
          </div>
          <div>
            <dt>Tip roll</dt>
            <dd className="num">{frame ? `${frame.tipRollDeg.toFixed(0)}°` : '—'}</dd>
          </div>
          <div>
            <dt>Grips</dt>
            <dd className="num">{frame ? frame.engageCount : '—'}</dd>
          </div>
        </dl>
      </div>
    </section>
  );
}
