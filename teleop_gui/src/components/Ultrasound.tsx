import { useEffect, useState } from 'react';
import type { UltrasoundFrame } from '../telemetry/types';
import styles from './Ultrasound.module.css';

interface Props {
  frame: UltrasoundFrame | null;
  /** Layout class from the parent — the panel does not decide where it sits. */
  className?: string;
  /** Milliseconds without a new picture before the panel calls the stream stale. */
  staleAfterMs?: number;
}

/**
 * Ultrasound monitor.
 *
 * Shows the picture the bridge sent, and nothing else. The bridge is not the
 * probe's client — `us_frame_node` is, because the probe accepts only one — so
 * what arrives here has already been fan-converted for reading. A session
 * records the raw polar frame; this panel is a display copy and never a source.
 *
 * A still image and a dead stream look identical, which is why the header
 * carries the frame counter and an age: a frozen sector with a rising age is a
 * dropped link, and a frozen sector with a rising counter is a probe that is
 * connected but not scanning.
 */
export function Ultrasound({ frame, className, staleAfterMs = 1500 }: Props) {
  // Age has to advance without new frames, so the panel re-renders on a timer
  // rather than only when a picture arrives.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const ageMs = frame ? now - frame.receivedAt : null;
  const stale = ageMs !== null && ageMs > staleAfterMs;

  return (
    <section className={className ? `plate ${className}` : 'plate'}>
      <div className="plate__head">
        <span className="plate__title">Ultrasound</span>
        <div className={styles.meta}>
          {frame ? (
            <>
              <span>{frame.display === 'fan' ? 'FAN' : 'POLAR'}</span>
              <span>
                {frame.width}×{frame.height}
              </span>
              <span>#{frame.seq}</span>
              <span className={stale ? styles.stale : undefined}>
                {ageMs === null ? '—' : `${(ageMs / 1000).toFixed(1)} s`}
              </span>
            </>
          ) : (
            <span className={styles.stale}>NO SIGNAL</span>
          )}
        </div>
      </div>
      <div className={styles.body}>
        {frame ? (
          <img
            className={styles.picture}
            src={`data:image/jpeg;base64,${frame.jpeg}`}
            alt="Ultrasound B-mode"
          />
        ) : (
          <p className={styles.empty}>
            No picture. The bridge relays <code>/us/image</code>, which comes from
            <code> us_frame_node</code> — start it with the probe profile and its access point
            connected.
          </p>
        )}
      </div>
    </section>
  );
}
