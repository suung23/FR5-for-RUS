import { useEffect, useState } from 'react';
import { subscribeEvents, type LogEvent } from '../store/events';
import styles from './EventLog.module.css';

interface Props {
  /** Milliseconds since the last telemetry frame, or null when none received. */
  ageMs: number | null;
  stale: boolean;
}

/**
 * Bottom event strip.
 *
 * Transitions only, newest first, with a timestamp and a source tag. Beside it
 * sits telemetry freshness, because "the last event was 40 seconds ago" means
 * something completely different depending on whether frames are still
 * arriving.
 *
 * Severity is a word in a bordered box; an alarm row inverts. No colour.
 */
export function EventLog({ ageMs, stale }: Props) {
  const events = useEventLog();

  return (
    <footer className={styles.strip}>
      <div className={styles.logHead}>
        <span className={styles.logTitle}>EVENT LOG</span>
        <span className={styles.logCount}>{events.length}</span>
      </div>

      <ol className={styles.list}>
        {events.length === 0 ? (
          <li className={styles.empty}>NO EVENTS</li>
        ) : (
          events.slice(0, 4).map((event) => (
            <li key={event.id} className={event.severity === 'alarm' ? styles.alarm : undefined}>
              <span className={`num ${styles.time}`}>{formatTime(event.at)}</span>
              <span className={styles.severity}>{severityWord(event)}</span>
              <span className={styles.source}>{event.source}</span>
              <span className={styles.text}>{event.text}</span>
            </li>
          ))
        )}
      </ol>

      <div className={styles.meta}>
        <span className={styles.metaItem}>
          <span className={styles.metaLabel}>TELEMETRY AGE</span>
          <span className={`num ${styles.metaValue} ${stale ? styles.metaStale : ''}`}>
            {ageMs === null ? '—' : `${(ageMs / 1000).toFixed(1)} s`}
          </span>
        </span>
        <span className={styles.research}>
          RESEARCH USE ONLY · NOT A MEDICAL DEVICE · NOT FOR DIAGNOSIS OR TREATMENT ·
          DISPLAY IS READ-ONLY AND DOES NOT COMMAND THE ROBOT
        </span>
      </div>
    </footer>
  );
}

function useEventLog(): LogEvent[] {
  const [events, setEvents] = useState<LogEvent[]>([]);
  useEffect(() => subscribeEvents(setEvents), []);
  return events;
}

function severityWord(event: LogEvent): string {
  return event.severity === 'alarm' ? 'ALM' : event.severity === 'warn' ? 'WRN' : 'INF';
}

function formatTime(at: number): string {
  return new Date(at).toLocaleTimeString('en-GB', { hour12: false });
}
