import { useCallback, useEffect, useRef, useState } from 'react';
import type { UltrasoundFrame } from '../telemetry/types';
import styles from './Ultrasound.module.css';

interface Props {
  frame: UltrasoundFrame | null;
  /** Layout class from the parent — the panel does not decide where it sits. */
  className?: string;
  /** Milliseconds without a new picture before the panel calls the stream stale. */
  staleAfterMs?: number;
  /** True while the parent is giving this panel the whole row. */
  expanded?: boolean;
  /** Ask the parent for the whole row, or give it back. Absent: no button. */
  onToggleExpand?(): void;
}

/** Zoom limits. 1 is fit-to-panel; past 8 the speckle is bigger than the anatomy. */
const MIN_SCALE = 1;
const MAX_SCALE = 8;

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
export function Ultrasound({ frame, className, staleAfterMs = 1500,
                             expanded = false, onToggleExpand }: Props) {
  // Age has to advance without new frames, so the panel re-renders on a timer
  // rather than only when a picture arrives.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const ageMs = frame ? now - frame.receivedAt : null;
  const stale = ageMs !== null && ageMs > staleAfterMs;

  // Display-only zoom. Nothing here reaches the probe or the robot: it changes
  // which pixels the operator can look at closely, not which pixels arrived.
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  // Panning past the edge would leave the operator staring at black with no way
  // to tell which way the picture went. At scale s the picture overhangs the
  // panel by (s-1)/2 on each side, and that is the whole travel.
  const clamp = useCallback((next: { x: number; y: number }, s: number) => {
    const box = bodyRef.current?.getBoundingClientRect();
    const limX = box ? (box.width * (s - 1)) / 2 : 0;
    const limY = box ? (box.height * (s - 1)) / 2 : 0;
    return {
      x: Math.max(-limX, Math.min(limX, next.x)),
      y: Math.max(-limY, Math.min(limY, next.y)),
    };
  }, []);

  const reset = useCallback(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  // Zoom toward the cursor, so the structure under the pointer stays under it.
  const onWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      if (!frame) return;
      event.preventDefault();
      const box = bodyRef.current?.getBoundingClientRect();
      if (!box) return;
      const cx = event.clientX - box.left - box.width / 2;
      const cy = event.clientY - box.top - box.height / 2;
      setScale((prev) => {
        const next = Math.min(
          MAX_SCALE,
          Math.max(MIN_SCALE, prev * (event.deltaY < 0 ? 1.15 : 1 / 1.15)),
        );
        setOffset((o) =>
          next === MIN_SCALE
            ? { x: 0, y: 0 }
            : clamp(
                { x: cx - ((cx - o.x) * next) / prev, y: cy - ((cy - o.y) * next) / prev },
                next,
              ),
        );
        return next;
      });
    },
    [clamp, frame],
  );

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (scale === MIN_SCALE) return;
      drag.current = { x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [offset.x, offset.y, scale],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const from = drag.current;
      if (!from) return;
      setOffset(
        clamp({ x: from.ox + (event.clientX - from.x), y: from.oy + (event.clientY - from.y) }, scale),
      );
    },
    [clamp, scale],
  );

  const onPointerUp = useCallback(() => {
    drag.current = null;
  }, []);

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
          {scale > 1 ? (
            <button type="button" className={styles.zoomTag} onClick={reset} title="Back to fit">
              {scale.toFixed(1)}× · RESET
            </button>
          ) : null}
          {onToggleExpand ? (
            <button
              type="button"
              className={styles.expand}
              onClick={onToggleExpand}
              aria-pressed={expanded}
              title={expanded ? 'Give the row back to the arm view' : 'Take the whole row'}
            >
              {expanded ? 'FIT' : 'EXPAND'}
            </button>
          ) : null}
        </div>
      </div>
      <div
        className={styles.body}
        ref={bodyRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={reset}
        style={{ cursor: scale > 1 ? 'grab' : 'default' }}
      >
        {frame ? (
          <img
            className={styles.picture}
            src={`data:image/jpeg;base64,${frame.jpeg}`}
            alt="Ultrasound B-mode"
            draggable={false}
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: 'center center',
            }}
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
