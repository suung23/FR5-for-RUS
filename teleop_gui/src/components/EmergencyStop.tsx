import { useEffect, useRef, useState } from 'react';
import type { EstopPhase, EstopSource, EstopState, RobotTelemetry } from '../telemetry/types';
import { JOINT_NAMES } from '../telemetry/fr5Model';
import styles from './EmergencyStop.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

/**
 * Emergency stop, protective stop, and the return to the initial pose.
 *
 * **The three levels are kept apart here, deliberately** (safety_estop/README §1):
 *
 * - *Emergency stop* is the physical latching button wired into the
 *   controller's dual-channel safety input. No software is on that path. This
 *   plate only **observes** it, and when the installation has no such circuit
 *   it says so in the hardware row rather than letting the word "ESTOP" on the
 *   phase line imply one exists.
 * - *Protective stop* is what this supervisor trips automatically: contact
 *   force over the threshold, any joint at 90 % of its travel, or a sensor
 *   going stale. It only works while the GUI, ROS, the supervisor and the
 *   sensors are all alive.
 * - *STOP* below is the operator's own command. Same fragile path.
 *
 * **The readback is the truth; every button here is a request.** Phase, the
 * latch and the thresholds are what the supervisor reports. A refused command
 * shows its reason and changes nothing — including a threshold the supervisor
 * would not take.
 *
 * RECOVER is held rather than clicked. It is the one control on this console
 * that starts a motion while the arm is in contact, and a motion that begins
 * on a mis-click next to a STOP button is a bad trade for one saved second.
 */

const PHASE_TEXT: Record<EstopPhase, { label: string; note: string }> = {
  disarmed: { label: 'DISARMED', note: 'Nothing is watching. Force and joint limits are not armed.' },
  armed: { label: 'ARMED', note: 'Watching force and joint travel. Ordinary motion is allowed.' },
  tripped: { label: 'TRIPPED', note: 'Motion inhibited. Read the cause, then recover deliberately.' },
  releasing: { label: 'RELEASING', note: 'Backing out along the path it came in, slowly, still in contact.' },
  returning: { label: 'RETURNING', note: 'Contact released. Retracing the recorded path to the initial pose.' },
  safe: { label: 'SAFE', note: 'At the initial pose. Motion is still inhibited until RESET.' },
};

const SOURCE_TEXT: Record<EstopSource, string> = {
  hardware: 'EMERGENCY STOP · physical circuit',
  force: 'Protective stop · contact force',
  joint: 'Protective stop · joint travel',
  force_stale: 'Protective stop · force sensor lost',
  joint_stale: 'Protective stop · joint state lost',
  operator: 'Operator stop · console button',
};

const HOLD_MS = 1500;

export function EmergencyStopPlate({ telemetry, available, onCommand }: Props) {
  const estop: EstopState | undefined = telemetry.estop;
  const stale = estop?.ageS !== undefined && estop.ageS > 2;

  const send = (command: string, extra: Record<string, unknown> = {}) =>
    onCommand({ command: `estop.${command}`, requestId: Date.now(), ...extra });

  if (!estop) {
    return (
      <section className="plate">
        <div className="plate__head">
          <span className="plate__title">Emergency stop · protective stop</span>
          <span className="plate__aside">no supervisor</span>
        </div>
        <div className={`plate__body ${styles.body}`}>
          <p className={styles.absent}>
            The safety supervisor has never been heard from.{' '}
            <strong>This is not the same as “no stop is in force.”</strong> Nothing on this
            console is watching contact force or joint travel, and nothing would stop the arm
            if either went out of bounds. Start it with{' '}
            <code>ros2 run safety_estop safety_estop</code>.
          </p>
        </div>
      </section>
    );
  }

  const phase = PHASE_TEXT[estop.phase];
  const trip = estop.latched;
  const hw = estop.hardware;
  const alarm = estop.phase === 'tripped' || estop.inhibit;
  const moving = estop.phase === 'releasing' || estop.phase === 'returning';

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Emergency stop · protective stop</span>
        <span className="plate__aside">
          {stale ? `supervisor silent ${estop.ageS?.toFixed(0)} s` : 'as the supervisor holds it'}
        </span>
      </div>

      <div className={`plate__body ${styles.body}`}>
        {/* Phase first and largest: it answers "can this arm move right now". */}
        <div className={`${styles.phase} ${alarm ? styles.phaseAlarm : ''}`}>
          <span className={styles.phaseLabel}>{phase.label}</span>
          <span className={styles.phaseNote}>{phase.note}</span>
          <span className={`${styles.inhibit} ${estop.inhibit ? styles.inhibitOn : ''}`}>
            {estop.inhibit ? 'MOTION INHIBITED' : 'MOTION ALLOWED'}
          </span>
        </div>

        {stale ? (
          <p className={styles.warnLine}>
            The supervisor has not reported for {estop.ageS?.toFixed(0)} s. Everything below is
            the last thing it said, not the current state. The servo node stops on its own when
            this channel dies.
          </p>
        ) : null}

        {/* The physical circuit. Its own row, never folded into the phase. */}
        <div className={styles.hardware}>
          <span className="field__label">Physical E-stop circuit</span>
          <span className={`${styles.hwValue} num ${hw?.present ? '' : styles.hwAbsent}`}>
            {!hw
              ? 'NOT REPORTED'
              : !hw.present
                ? 'NOT INSTALLED'
                : hw.channelsDisagree
                  ? 'CHANNEL FAULT'
                  : hw.stale
                    ? 'UNREADABLE'
                    : hw.asserted
                      ? 'PRESSED'
                      : 'CLEAR'}
          </span>
        </div>
        {hw?.reason ? <p className={styles.hwReason}>{hw.reason}</p> : null}

        {/* What tripped. */}
        {trip ? (
          <div className={styles.trip}>
            <span className={styles.tripSource}>{SOURCE_TEXT[trip.source] ?? trip.source}</span>
            <span className={styles.tripDetail}>{trip.detail}</span>
            {!trip.allowsRelease ? (
              <span className={styles.tripBlocked}>
                Automatic recovery is not permitted for this cause — the condition that would
                tell the robot when to stop backing out is the one that failed. Free the arm by
                hand, fix the cause, then RESET.
              </span>
            ) : null}
          </div>
        ) : null}

        {estop.haltReason ? (
          <p className={styles.warnLine}>Recovery stopped short: {estop.haltReason}</p>
        ) : null}

        {/* How close the live values are to the thresholds. */}
        <Margins estop={estop} />

        {/* The return path. */}
        <div className={styles.trail}>
          <span className="field__label">Return path</span>
          <span className={`${styles.trailValue} num`}>
            {estop.trail.points === 0
              ? '— not recording'
              : `${estop.trail.points} pts` +
                (estop.trail.remaining !== null ? ` · ${estop.trail.remaining} left` : '') +
                ` · ${estop.recovery.releaseSpeedDegS.toFixed(0)}→${estop.recovery.returnSpeedDegS.toFixed(0)} °/s`}
          </span>
        </div>
        {estop.trail.truncated ? (
          <p className={styles.warnLine}>
            The oldest waypoints have been dropped. “Initial pose” now means the oldest one
            still held, not where the session started.
          </p>
        ) : null}

        <Thresholds estop={estop} available={available} send={send} />

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.stop}
            disabled={!available}
            onClick={() => send('stop', { detail: 'console STOP' })}
          >
            STOP
          </button>
          <button
            type="button"
            disabled={!available || estop.phase !== 'armed'}
            onClick={() => send('disarm')}
          >
            Disarm
          </button>
          <button
            type="button"
            disabled={!available || !(estop.phase === 'disarmed' || estop.phase === 'safe')}
            onClick={() => send('arm')}
          >
            Arm
          </button>
          {moving ? (
            <button type="button" className={styles.abort} onClick={() => send('abort')}>
              Abort return
            </button>
          ) : (
            <HoldButton
              label="Hold to return home"
              disabled={!available || estop.phase !== 'tripped' || !(trip?.allowsRelease ?? false)}
              onComplete={() => send('recover')}
            />
          )}
          <button
            type="button"
            disabled={!available || trip === null || moving}
            onClick={() => send('reset')}
          >
            Reset
          </button>
        </div>

        {estop.lastCommand && !estop.lastCommand.ok ? (
          <p className={styles.refusal}>
            <strong>{estop.lastCommand.command}</strong> refused — {estop.lastCommand.reason}
          </p>
        ) : null}

        <p className={styles.note}>
          RESET clears the latch and leaves the supervisor <em>disarmed</em>; arming again is a
          separate press, so a reset can never be a restart. Recovery retraces the joint path
          recorded since ARM rather than driving a straight line home — the straight line&apos;s
          first metres are through whatever the probe is pressed against.
        </p>
      </div>
    </section>
  );
}

/** Live values against their thresholds — the thing that tells you a trip is coming. */
function Margins({ estop }: { estop: EstopState }) {
  const force = estop.forceN;
  const forceFrac = force === null ? 0 : Math.min(1, Math.abs(force) / estop.limits.forceTripN);
  const worst = estop.worstJoint;

  return (
    <div className={styles.margins}>
      <Bar
        label="Contact force"
        value={force === null ? '—' : `${Math.abs(force).toFixed(2)} / ${estop.limits.forceTripN.toFixed(1)} N`}
        fraction={forceFrac}
        trip={1}
      />
      <div className={styles.joints}>
        {estop.jointReach.map((reach, i) => (
          <div key={JOINT_NAMES[i] ?? i} className={styles.joint}>
            <span className={styles.jointName}>{JOINT_NAMES[i] ?? `J${i + 1}`}</span>
            <span className={styles.jointTrack}>
              <span
                className={`${styles.jointFill} ${
                  reach >= estop.limits.jointReachTrip ? styles.jointFillTrip : ''
                }`}
                style={{ width: `${Math.min(100, reach * 100)}%` }}
              />
              <span
                className={styles.jointMark}
                style={{ left: `${estop.limits.jointReachTrip * 100}%` }}
              />
            </span>
            <span className={`${styles.jointMargin} num ${i === worst ? styles.worst : ''}`}>
              {estop.jointMarginDeg[i] !== undefined
                ? `${estop.jointMarginDeg[i].toFixed(0)}°`
                : '—'}
            </span>
          </div>
        ))}
      </div>
      <p className={styles.marginNote}>
        Joint bars show travel used from mid-range toward the bound being approached; the tick
        is the {Math.round(estop.limits.jointReachTrip * 100)} % trip line and the number is the
        degrees still left to that bound.
      </p>
    </div>
  );
}

function Bar({
  label,
  value,
  fraction,
  trip,
}: {
  label: string;
  value: string;
  fraction: number;
  trip: number;
}) {
  return (
    <div className={styles.bar}>
      <span className={styles.barLabel}>{label}</span>
      <span className={styles.barTrack}>
        <span
          className={`${styles.barFill} ${fraction >= trip ? styles.barFillTrip : ''}`}
          style={{ width: `${Math.min(100, fraction * 100)}%` }}
        />
      </span>
      <span className={`${styles.barValue} num`}>{value}</span>
    </div>
  );
}

/**
 * The two thresholds the operator may change.
 *
 * The ceiling and the confirm windows are not here: they live in `estop.yaml`,
 * because the easiest way out of a trip must not be a control on the same
 * screen as the trip. The supervisor refuses a threshold change while a trip is
 * latched for the same reason.
 */
function Thresholds({
  estop,
  available,
  send,
}: {
  estop: EstopState;
  available: boolean;
  send(command: string, extra?: Record<string, unknown>): boolean;
}) {
  const [forceText, setForceText] = useState('');
  const [reachText, setReachText] = useState('');
  const [seeded, setSeeded] = useState(false);

  useEffect(() => {
    if (seeded) return;
    setForceText(estop.limits.forceTripN.toFixed(1));
    setReachText((estop.limits.jointReachTrip * 100).toFixed(0));
    setSeeded(true);
  }, [estop.limits, seeded]);

  const force = Number(forceText);
  const reachPct = Number(reachText);
  const ceiling = estop.limits.forceCeilingN;
  const reachCeiling = estop.limits.jointReachCeiling * 100;

  let problem: string | null = null;
  if (!Number.isFinite(force) || force <= 0) problem = 'Force threshold must be a positive number.';
  else if (force > ceiling) problem = `Force threshold cannot exceed the ${ceiling.toFixed(0)} N ceiling.`;
  else if (force > estop.limits.forceInstantN)
    problem = `Must stay under the instant-trip value (${estop.limits.forceInstantN.toFixed(0)} N).`;
  else if (!Number.isFinite(reachPct) || reachPct <= 0 || reachPct > reachCeiling)
    problem = `Joint threshold must be between 1 and ${reachCeiling.toFixed(0)} %.`;
  else if (estop.latched !== null)
    problem = 'Thresholds cannot change while a stop is latched — clear it first.';

  const dirty =
    Math.abs(force - estop.limits.forceTripN) > 1e-6 ||
    Math.abs(reachPct / 100 - estop.limits.jointReachTrip) > 1e-6;

  return (
    <div className={styles.thresholds}>
      <label className={styles.entry}>
        <span className={styles.entryLabel}>Force trip</span>
        <span className={styles.entryRow}>
          <input
            className={`${styles.input} num`}
            type="number"
            step={0.5}
            min={0.5}
            max={ceiling}
            value={forceText}
            disabled={!available}
            onChange={(e) => setForceText(e.target.value)}
          />
          <span className={styles.unit}>N</span>
        </span>
      </label>
      <label className={styles.entry}>
        <span className={styles.entryLabel}>Joint trip</span>
        <span className={styles.entryRow}>
          <input
            className={`${styles.input} num`}
            type="number"
            step={1}
            min={1}
            max={reachCeiling}
            value={reachText}
            disabled={!available}
            onChange={(e) => setReachText(e.target.value)}
          />
          <span className={styles.unit}>%</span>
        </span>
      </label>
      <span className={styles.thresholdNote}>
        Instant {estop.limits.forceInstantN.toFixed(0)} N · ceiling {ceiling.toFixed(0)} N ·
        {estop.limits.tripOnStale ? ' stale sensor trips' : ' stale sensor ignored'}
      </span>
      {problem ? <span className={styles.reason}>{problem}</span> : null}
      <button
        type="button"
        disabled={!available || problem !== null || !dirty}
        onClick={() =>
          send('limits', { forceTripN: round(force), jointReachTrip: round(reachPct) / 100 })
        }
      >
        Apply
      </button>
    </div>
  );
}

/**
 * A button that must be held.
 *
 * Used for the one control here that commands motion. The fill is the whole
 * feedback: it shows that holding is what is wanted, and releasing early
 * cancels with nothing sent.
 */
function HoldButton({
  label,
  disabled,
  onComplete,
}: {
  label: string;
  disabled: boolean;
  onComplete(): void;
}) {
  const [progress, setProgress] = useState(0);
  const start = useRef<number | null>(null);
  const frame = useRef<number | null>(null);

  const stop = () => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    start.current = null;
    setProgress(0);
  };

  useEffect(() => stop, []);

  const tick = () => {
    if (start.current === null) return;
    const elapsed = performance.now() - start.current;
    if (elapsed >= HOLD_MS) {
      stop();
      onComplete();
      return;
    }
    setProgress(elapsed / HOLD_MS);
    frame.current = requestAnimationFrame(tick);
  };

  return (
    <button
      type="button"
      className={styles.hold}
      disabled={disabled}
      onPointerDown={() => {
        if (disabled) return;
        start.current = performance.now();
        frame.current = requestAnimationFrame(tick);
      }}
      onPointerUp={stop}
      onPointerLeave={stop}
      onKeyDown={(e) => {
        if (e.key === ' ' || e.key === 'Enter') e.preventDefault();
      }}
    >
      <span className={styles.holdFill} style={{ width: `${progress * 100}%` }} />
      <span className={styles.holdLabel}>{label}</span>
    </button>
  );
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
