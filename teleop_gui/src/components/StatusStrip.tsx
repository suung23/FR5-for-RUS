import { useEffect, useState } from 'react';
import { config, isSimulated } from '../telemetry/config';
import { ROBOT_STATE_LABEL } from '../store/selectors';
import type { LinkStatus, RobotState, WrenchSource } from '../telemetry/types';
import styles from './StatusStrip.module.css';

interface Props {
  link: LinkStatus | null;
  linkUp: boolean;
  stale: boolean;
  sensorSource: WrenchSource | undefined;
  robotState: RobotState | undefined;
  frameRateHz: number;
  paused: boolean;
  onTogglePause(): void;
  onReset(): void;
}

/**
 * Top status strip.
 *
 * A row of explicit named fields, in the order an operator checks them:
 * system, robot link, sensor link, mode. Each is a label and a word — no
 * badges, no dots standing in for text. The strip is the one black surface in
 * the console, which separates the machine's own status from the white plates
 * that carry measurements.
 */
export function StatusStrip({
  link,
  linkUp,
  stale,
  sensorSource,
  robotState,
  frameRateHz,
  paused,
  onTogglePause,
  onReset,
}: Props) {
  const clock = useClock();

  const robotText = !linkUp ? 'DISCONNECTED' : stale ? 'STALE' : 'CONNECTED';
  // Name the sensor rather than saying CONNECTED. On this cell the important
  // distinction is which path the force came in on, not whether something did.
  const sensorText = !linkUp
    ? 'DISCONNECTED'
    : sensorSource === 'px6d_serial'
      ? 'PX6D USB'
      : sensorSource === 'controller'
        ? 'CONTROLLER'
        : sensorSource === 'simulation'
          ? 'SIMULATED'
          : 'NO DATA';
  const modeText = !linkUp || !robotState ? 'UNKNOWN' : ROBOT_STATE_LABEL[robotState].toUpperCase();

  return (
    <header className={styles.strip}>
      <span className={styles.system}>FR5 · RUS CONSOLE</span>

      <Field label="ROBOT" value={robotText} bad={!linkUp || stale} />
      <Field
        label="SENSOR"
        value={sensorText}
        bad={!linkUp || sensorSource === undefined || sensorSource === 'none'}
      />
      <Field label="MODE" value={modeText} />
      <Field
        label="SOURCE"
        value={isSimulated ? 'SIMULATION' : (link?.transport ?? config.transport).toUpperCase()}
        bad={isSimulated}
      />

      <span className={styles.spacer} />

      <span className={styles.meta}>
        <span className="num">{frameRateHz}</span> Hz
      </span>
      <span className={`num ${styles.clock}`}>{clock}</span>

      <div className={styles.actions}>
        <button type="button" onClick={onTogglePause} aria-pressed={paused}>
          {paused ? 'Resume' : 'Hold'}
        </button>
        <button type="button" onClick={onReset}>
          Reset
        </button>
      </div>
    </header>
  );
}

function Field({ label, value, bad }: { label: string; value: string; bad?: boolean }) {
  return (
    <span className={styles.field}>
      <span className={styles.fieldLabel}>{label}</span>
      <span className={`${styles.fieldValue} ${bad ? styles.fieldValueBad : ''}`}>{value}</span>
    </span>
  );
}

function useClock(): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toLocaleTimeString('en-GB', { hour12: false });
}
