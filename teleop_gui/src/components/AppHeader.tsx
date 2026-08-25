import { useEffect, useState } from 'react';
import { config, isSimulated } from '../telemetry/config';
import type { LinkStatus } from '../telemetry/types';
import styles from './AppHeader.module.css';

interface Props {
  link: LinkStatus | null;
  connected: boolean;
  stale: boolean;
  frameRateHz: number;
  paused: boolean;
  onTogglePause(): void;
  onReset(): void;
}

const PHASE_TEXT: Record<LinkStatus['phase'], string> = {
  idle: 'Idle',
  connecting: 'Connecting',
  connected: 'Live',
  reconnecting: 'Reconnecting',
  error: 'Link error',
  closed: 'Closed',
};

export function AppHeader({
  link,
  connected,
  stale,
  frameRateHz,
  paused,
  onTogglePause,
  onReset,
}: Props) {
  const clock = useClock();

  const severity = !connected || stale ? 'critical' : link?.phase === 'connected' ? 'nominal' : 'caution';
  const phase = stale && link?.phase === 'connected' ? 'Stale' : PHASE_TEXT[link?.phase ?? 'idle'];

  return (
    <header className={styles.header}>
      <div className={styles.identity}>
        <div className={styles.mark} aria-hidden="true">
          <span className={styles.markRing} />
          <span className={styles.markCore} />
        </div>
        <div>
          <h1 className={styles.title}>Robotic Ultrasound Teleoperation Monitor</h1>
          <p className={styles.subtitle}>
            FR5 right arm · PX6D six-axis force/torque · monitoring only
          </p>
        </div>
      </div>

      <div className={styles.research} role="note">
        <span className={styles.researchTag}>Research use only</span>
        <span className={styles.researchText}>
          Not a medical device. Not cleared or approved for diagnosis or treatment.
          Display is read-only and does not command the robot.
        </span>
      </div>

      <div className={styles.link}>
        <div className={styles.linkRow}>
          <span className={`dot dot--${severity}`} />
          <span className={styles.linkPhase}>{phase}</span>
          <span className={styles.linkKind}>{link?.transport ?? config.transport}</span>
        </div>
        <div className={styles.linkMeta}>
          <span className="num">{link?.endpoint ?? '—'}</span>
          {isSimulated ? null : <span className={styles.host}>host {config.robotHost}</span>}
        </div>
        {link?.error ? <div className={styles.linkError}>{link.error}</div> : null}
      </div>

      <div className={styles.controls}>
        <div className={styles.rate}>
          <span className="num">{frameRateHz.toString().padStart(2, '0')}</span>
          <span className={styles.rateUnit}>Hz</span>
        </div>
        <span className={`num ${styles.clock}`}>{clock}</span>
        <button type="button" onClick={onTogglePause} aria-pressed={paused}>
          {paused ? 'Resume' : 'Hold'}
        </button>
        <button type="button" onClick={onReset}>
          Reset session
        </button>
      </div>
    </header>
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
