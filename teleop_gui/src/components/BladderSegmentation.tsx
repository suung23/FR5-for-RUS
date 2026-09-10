import { useEffect, useRef, useState } from 'react';
import type { SegmentationFrame } from '../telemetry/types';
import styles from './BladderSegmentation.module.css';

interface Props {
  frame: SegmentationFrame | null;
  className?: string;
  /** Milliseconds without a new frame before the panel calls the stream stale. */
  staleAfterMs?: number;
  /** Above this, the verdict beside the picture is not the verdict *for* it. */
  stateAgeWarnMs?: number;
}

/**
 * Bladder segmentation monitor.
 *
 * Shows the frame the U-Net was fed and the mask it returned, and says how much
 * that mask can be trusted. Nothing here is computed from the picture: every
 * number comes from the `ControlState` the network's own feature extractor
 * produced, relayed unchanged through `run_segmentation` and the bridge.
 *
 * **The base picture is not the Ultrasound panel's sector.** That one is
 * `fr5_vision.scan_convert`'s fan; this one is `rus_policy.bmode`'s 256²
 * letterbox, which is the input the network actually sees. They are different
 * scan conversions, so a mask drawn over the other picture would sit in the
 * wrong place and look entirely convincing. The pair travels together and is
 * matched by ROS header stamp in the bridge before it is sent.
 *
 * Three things the operator can turn off, because each is a claim that has to
 * be checkable:
 *
 * * **Mask** — blink it off. A boundary that follows the lumen edge underneath
 *   is a mask; one that stays put while the edge moves is a model repeating
 *   itself. There is no other way to see that from a still overlay.
 * * **Fill** — the tint is easy to read and hides the speckle it covers.
 *   Outline-only leaves the interior visible.
 * * **Marks** — the centroid and the beam axis. `ê` is measured between them,
 *   and the beam axis is the centre of the **imaged sector**, not of the frame.
 *
 * `Q_seg` presupposes the bladder was found. When there is no mask the panel
 * prints NO MASK rather than a score, because a number there would be read as
 * "poor image" when the fact is "nothing to score".
 */

/** A mask pixel is inside when the relayed PNG says 255. It is binary by construction. */
const INSIDE = 128;

function pct(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)} %`;
}

function num(value: number | null | undefined, digits = 2, unit = ''): string {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)}${unit}`;
}

/** Decode a base64 image the bridge sent. Rejects rather than drawing a blank. */
function load(base64: string, mime: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('decode failed'));
    img.src = `data:${mime};base64,${base64}`;
  });
}

/** A theme token, resolved now. Follows the contact-probing palette swap. */
function token(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * `#rrggbb` → `[r, g, b]`. Only the hex form is handled because that is the
 * only form the palette is written in; anything else falls back rather than
 * producing a colour nobody chose.
 */
function rgb(hex: string, fallback: [number, number, number]): [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!match) return fallback;
  const n = parseInt(match[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function BladderSegmentation({
  frame,
  className,
  staleAfterMs = 1500,
  stateAgeWarnMs = 500,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showMask, setShowMask] = useState(true);
  const [showFill, setShowFill] = useState(true);
  const [showMarks, setShowMarks] = useState(true);
  const [drawError, setDrawError] = useState<string | null>(null);

  // Age has to advance without new frames, so the panel re-renders on a timer
  // rather than only when one arrives.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    let cancelled = false;

    void (async () => {
      let base: HTMLImageElement;
      let mask: HTMLImageElement;
      try {
        [base, mask] = await Promise.all([
          load(frame.jpeg, 'image/jpeg'),
          load(frame.mask, 'image/png'),
        ]);
      } catch {
        if (!cancelled) setDrawError('frame did not decode');
        return;
      }
      if (cancelled) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        setDrawError('no 2d context');
        return;
      }
      setDrawError(null);

      // Draw in image pixels. The canvas is scaled by CSS, so what is on screen
      // is exactly the pixels the network classified, at whatever size the
      // window allows — no resampling invents a boundary between them.
      const w = base.naturalWidth || frame.width || 256;
      const h = base.naturalHeight || frame.height || 256;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(base, 0, 0, w, h);

      if (showMask && mask.naturalWidth === w && mask.naturalHeight === h) {
        const scratch = document.createElement('canvas');
        scratch.width = w;
        scratch.height = h;
        const sctx = scratch.getContext('2d', { willReadFrequently: true });
        if (sctx) {
          sctx.drawImage(mask, 0, 0);
          const src = sctx.getImageData(0, 0, w, h).data;
          const out = sctx.createImageData(w, h);
          const px = out.data;
          const [r, g, b] = rgb(token('--accent', '#74b184'), [116, 177, 132]);

          for (let y = 0; y < h; y += 1) {
            for (let x = 0; x < w; x += 1) {
              const i = (y * w + x) * 4;
              if (src[i] < INSIDE) continue;
              // A pixel on the edge of the frame counts as a boundary pixel: the
              // mask is cut off there, and that is exactly what
              // `border_contact_ratio` is warning about.
              const edge =
                x === 0 ||
                y === 0 ||
                x === w - 1 ||
                y === h - 1 ||
                src[i - 4] < INSIDE ||
                src[i + 4] < INSIDE ||
                src[i - w * 4] < INSIDE ||
                src[i + w * 4] < INSIDE;
              px[i] = r;
              px[i + 1] = g;
              px[i + 2] = b;
              px[i + 3] = edge ? 255 : showFill ? 56 : 0;
            }
          }
          sctx.putImageData(out, 0, 0);
          // Through `drawImage`, not `putImageData` — the latter replaces the
          // destination instead of blending, which would erase the B-mode under
          // the mask and leave a flat shape with nothing to check it against.
          ctx.drawImage(scratch, 0, 0);
        }
      }

      const state = frame.state;
      if (showMarks && state) {
        ctx.lineWidth = 1;
        if (state.beamAxisPx !== null) {
          // The axis `ê` is measured from. Dashed so it cannot be mistaken for
          // anything the network drew.
          ctx.strokeStyle = rgba(token('--ink', '#e9ede9'), 0.55);
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(state.beamAxisPx + 0.5, 0);
          ctx.lineTo(state.beamAxisPx + 0.5, h);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (state.centroidPx) {
          const [cx, cy] = state.centroidPx;
          ctx.strokeStyle = rgba(token('--accent', '#74b184'), 1);
          ctx.beginPath();
          ctx.moveTo(cx - 6, cy + 0.5);
          ctx.lineTo(cx + 6, cy + 0.5);
          ctx.moveTo(cx + 0.5, cy - 6);
          ctx.lineTo(cx + 0.5, cy + 6);
          ctx.stroke();
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [frame, showMask, showFill, showMarks]);

  const ageMs = frame ? now - frame.receivedAt : null;
  const stale = ageMs !== null && ageMs > staleAfterMs;
  const state = frame?.state;
  const mismatched =
    frame?.stateAgeMs !== undefined && frame.stateAgeMs > stateAgeWarnMs;

  return (
    <section className={className ? `plate ${className}` : 'plate'}>
      <div className="plate__head">
        <span className="plate__title">Bladder segmentation</span>
        <div className={styles.meta}>
          {frame ? (
            <>
              <span>
                {frame.width}×{frame.height}
              </span>
              <span>#{frame.seq}</span>
              <span>{num(state?.perceptionMs, 0, ' ms')}</span>
              <span className={stale ? styles.stale : undefined}>
                {ageMs === null ? '—' : `${(ageMs / 1000).toFixed(1)} s`}
              </span>
            </>
          ) : (
            <span className={styles.stale}>NOT RUNNING</span>
          )}
        </div>
      </div>

      {frame ? (
        <>
          <div className={styles.controls} role="group" aria-label="Overlay">
            <button
              type="button"
              aria-pressed={showMask}
              onClick={() => setShowMask((v) => !v)}
              title="Blink the mask off to check the boundary against the lumen edge"
            >
              Mask
            </button>
            <button
              type="button"
              aria-pressed={showFill}
              disabled={!showMask}
              onClick={() => setShowFill((v) => !v)}
              title="Outline only leaves the speckle inside the lumen visible"
            >
              Fill
            </button>
            <button
              type="button"
              aria-pressed={showMarks}
              onClick={() => setShowMarks((v) => !v)}
              title="Centroid and beam axis — ê is measured between them"
            >
              Marks
            </button>
            <span className={styles.verdict}>
              {!state ? (
                <span className="tag tag--off">NO STATE</span>
              ) : !state.hasMask ? (
                <span className="tag tag--critical">NO MASK</span>
              ) : state.validForControl ? (
                <span className="tag tag--green">VALID</span>
              ) : (
                <span className="tag tag--strong">NOT VALID</span>
              )}
            </span>
          </div>

          <div className={styles.body}>
            <canvas ref={canvasRef} className={styles.picture} />
          </div>

          {drawError ? <p className={styles.note}>{drawError}</p> : null}

          {mismatched ? (
            <p className={styles.note}>
              The verdict below is {frame.stateAgeMs} ms older than this picture — they may
              not be the same frame.
            </p>
          ) : null}

          <div className={styles.readout}>
            <div className="field">
              <span className="field__label">Q_seg</span>
              <span className="field__value">
                {state?.hasMask ? num(state.quality, 3) : 'no mask'}
              </span>
            </div>
            <div className="field">
              <span className="field__label">Area (ROI)</span>
              <span className="field__value">{pct(state?.maskAreaRatio)}</span>
            </div>
            <div className="field">
              <span className="field__label">ê</span>
              <span className="field__value">
                {num(state?.eHatPx, 1, ' px')}
                {state?.token ? `  ${state.token}` : ''}
              </span>
            </div>
            <div className="field">
              <span className="field__label">Confidence</span>
              <span className="field__value">{num(state?.segmentationConfidence, 3)}</span>
            </div>
            <div className="field">
              <span className="field__label">Lumen contrast</span>
              <span className="field__value">{num(state?.lumenContrast, 3)}</span>
            </div>
            <div className="field">
              <span className="field__label">Temporal IoU</span>
              <span className="field__value">{num(state?.temporalWarpedIou, 3)}</span>
            </div>
            {state && state.rejectionReasons.length > 0 ? (
              <div className="field field--warn">
                <span className="field__label">Rejected</span>
                <span className="field__value">{state.rejectionReasons.join(', ')}</span>
              </div>
            ) : null}
            <div className="field">
              <span className="field__label">Checkpoint</span>
              <span className="field__value">{state?.checkpointId ?? '—'}</span>
            </div>
          </div>
        </>
      ) : (
        <div className={styles.body}>
          <p className={styles.empty}>
            No segmentation. The bridge relays <code>/us/seg/*</code>, which comes from{' '}
            <code>run_segmentation.py</code> — it is not part of the session script. Source ROS
            first, then <code>Unet_seg/.venv</code>, then:
            <br />
            <code>python3 policy_learning/scripts/run_segmentation.py</code>
          </p>
        </div>
      )}
    </section>
  );
}

/** `#rrggbb` at an alpha, for canvas strokes. */
function rgba(hex: string, alpha: number): string {
  const [r, g, b] = rgb(hex, [233, 237, 233]);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
