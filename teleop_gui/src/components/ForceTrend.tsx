import { memo, useMemo, useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { config } from '../telemetry/config';
import { contactMagnitude, type ForcePoint } from '../store/telemetryStore';
import styles from './ForceTrend.module.css';
import { usePalette } from '../telemetry/theme';

interface Props {
  history: ForcePoint[];
  /**
   * Bins per second behind the trace, or null when each point is one reading.
   *
   * The plate says which it is. A line drawn from 10 ms bins of a 1 kHz stream
   * and a line drawn at one point a second look alike at a glance and mean
   * very different things, and the operator should not have to guess which
   * they are reading.
   */
  waveformHz: number | null;
}

type Series = 'normal' | 'components';
type Scale = 'fit' | 'full';

const AXIS_STYLE = { fontSize: 9.5, fill: '#4a4a4a', fontFamily: 'Arial, Helvetica, sans-serif' };

/**
 * Narrowest window the fitted scale will open to, in newtons.
 *
 * Chosen so the gridlines land on **0.1 N per division**: the tick step is
 * picked near `span / 6`, and 0.6 is the widest span that still rounds down to
 * 0.1, giving six even divisions. Reading a value off the plot is then
 * arithmetic the operator does not have to do.
 *
 * The plot is about 230 px tall. Across the full 0–18 N range a 0.01 N change
 * is an eighth of a pixel — not small, *absent*; at this floor it is about
 * 4 px, still comfortably visible and about a quarter of the 0.016 N noise a
 * single 10 ms bin carries.
 *
 * The floor also stops a quiet signal being magnified until sensor noise fills
 * the plate and reads as motion. 0.6 N is twelve times the PX6D's ±0.05 N
 * per-axis noise, so noise stays a band near the line rather than the picture.
 *
 * Above this the step ladder takes over — a signal that genuinely needs more
 * than 0.6 N moves to 0.2 N per division, and so on. The axis says its span in
 * the header either way.
 */
const MIN_SPAN_N = 0.6;

/**
 * Most points handed to the chart.
 *
 * The buffer holds twenty seconds at the bridge's bin rate — two thousand
 * points at 100 Hz — and the plot is about a thousand pixels wide. Handing
 * recharts every point made it rebuild two SVG paths of that length on every
 * frame the bridge delivered, which measured at 6 fps on the monitoring view
 * against 17 fps with the plot off.
 *
 * Groups are folded, never sampled — the same rule as the bridge's bins. The
 * group mean carries the line and the group's own min/max widen the band, so
 * an excursion inside a group survives the reduction rather than depending on
 * which point happened to be picked. Sampling here would alias exactly the
 * detail the 100 Hz stream exists to show.
 *
 * This changes only what is drawn. The store keeps every point.
 */
const MAX_DRAWN_POINTS = 400;

/**
 * A round step near `span / 6`, so the axis carries readable numbers.
 *
 * Six divisions rather than four because the range is snapped outward to whole
 * steps: a coarse step wastes plot height on empty margin. At four, a 5.3 N
 * range rounded out to 8 N and threw away a third of the sensitivity that
 * fitting the axis was for.
 */
function niceStep(span: number): number {
  const raw = Math.max(span, 1e-6) / 6;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalised = raw / magnitude;
  const step = normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 5 ? 5 : 10;
  return step * magnitude;
}

/**
 * Snap a data range out to whole steps.
 *
 * A axis recomputed from the data every flush would slide continuously as
 * points enter and leave the 20 s window, and a trace that moves because the
 * axis moved is unreadable — the eye cannot tell which of the two changed.
 * Quantising to the tick step means the scale holds still until the signal
 * genuinely leaves it, and then jumps once.
 */
/**
 * Snap a data range out to whole steps, returning the step it used.
 *
 * The step **must** travel with the range. Snapping widens the span by up to
 * one step at each end, and re-deriving the step from the widened result gives
 * a different, coarser answer than the one the range was built on — a 0.6 N
 * window snapped to 0.7 came back asking for 0.2 N gridlines instead of the
 * 0.1 it had just been laid out for.
 */
function fitRange(
  low: number,
  high: number,
  minSpan: number,
): { domain: [number, number]; step: number } {
  const span = Math.max(high - low, minSpan);
  const middle = (low + high) / 2;
  const step = niceStep(span);
  return {
    domain: [
      Math.floor((middle - span / 2) / step) * step,
      Math.ceil((middle + span / 2) / step) * step,
    ],
    step,
  };
}

function ticksFor([low, high]: [number, number], step: number): number[] {
  const out: number[] = [];
  for (let v = low; v <= high + step / 2 && out.length < 16; v += step) {
    out.push(Number(v.toFixed(6)));
  }
  return out;
}

/** Enough decimals that neighbouring ticks do not print the same number. */
function decimalsFor(step: number): number {
  if (step >= 1) return 0;
  if (step >= 0.1) return 1;
  if (step >= 0.01) return 2;
  return 3;
}

/**
 * Rolling force trend.
 *
 * Two series, chosen by a tab rather than crammed together. `CONTACT` answers
 * the operating question — how much margin is left to the limit — with the
 * warn and limit rules drawn on the plot. It carries the same `‖F‖` the
 * headline plate and the control stack act on, so the shape of the trace and
 * the number above it are one quantity; a trend in a different scalar from the
 * reading it explains is a trend that will eventually contradict it.
 *
 * `COMPONENTS` stays on the **raw sensor channels** on purpose. It is the
 * diagnostic for the axis assignment, which is still provisional, and that
 * check needs what the sensor actually reported rather than what compensation
 * made of it.
 *
 * Traces are black; only the measured contact force is navy, so the line the
 * operator is actually reading is the one that differs.
 */
function ForceTrendView({ history, waveformHz }: Props) {
  // Recharts takes colours as props, not from CSS, so the trace has to be told
  // which palette is in force. `accent` is the structural colour — green at
  // rest, blue in contact probing — and the trace, the hold band and the grid
  // all follow it so the plot does not stay green inside a blue console.
  const palette = usePalette();
  const accent = palette['--green'];
  const rule = palette['--rule'];
  const [series, setSeries] = useState<Series>('normal');
  // Fitted by default. The gauge on the status column already carries absolute
  // margin to the limit, with its own warn and limit marks and its OVER LIMIT
  // tag — so the trend does not have to, and is free to specialise in the
  // structure of the force instead of its size. `FULL` puts the limit back on
  // the plot for anyone who wants both in one place.
  const [scale, setScale] = useState<Scale>('fit');

  const data = useMemo(() => {
    if (history.length === 0) return [];
    const now = history[history.length - 1].t;
    const stride = Math.ceil(history.length / MAX_DRAWN_POINTS);

    // `band` is the pair recharts draws a range area from. Built here rather
    // than read as two keys because a range series takes one key holding both
    // edges, and building it in the view keeps the store's point a plain record.
    const point = (p: ForcePoint, t: number) => {
      const fc = contactMagnitude(p);
      const a = contactMagnitude(p, p.czLo);
      const b = contactMagnitude(p, p.czHi);
      return {
        t,
        age: (t - now) / 1000,
        fc,
        fcLo: Math.min(a, b),
        fcHi: Math.max(a, b),
        fx: p.fx,
        fy: p.fy,
        fz: p.fz,
        band: [Math.min(a, b), Math.max(a, b)] as [number, number],
      };
    };

    if (stride <= 1) return history.map((p) => point(p, p.t));

    const out: ReturnType<typeof point>[] = [];
    for (let i = 0; i < history.length; i += stride) {
      const end = Math.min(i + stride, history.length);
      const n = end - i;
      // The group is folded in the vector, then measured — the same order the
      // single-point path uses, so the zero comes off before the magnitude.
      let cx = 0;
      let cy = 0;
      let cz = 0;
      let fx = 0;
      let fy = 0;
      let fz = 0;
      let czLo = Infinity;
      let czHi = -Infinity;
      for (let k = i; k < end; k += 1) {
        const p = history[k];
        cx += p.cx;
        cy += p.cy;
        cz += p.cz;
        fx += p.fx;
        fy += p.fy;
        fz += p.fz;
        czLo = Math.min(czLo, p.czLo);
        czHi = Math.max(czHi, p.czHi);
      }
      out.push(
        point(
          { t: 0, cx: cx / n, cy: cy / n, cz: cz / n, czLo, czHi,
            fx: fx / n, fy: fy / n, fz: fz / n },
          history[i + (n >> 1)].t,
        ),
      );
    }
    return out;
  }, [history]);

  const banded = waveformHz !== null;

  // Scanned in a loop rather than spread into `Math.max`. At the bridge's
  // default the buffer is a couple of thousand points, and a waveform rate set
  // higher would push a spread past the engine's argument limit — a crash that
  // would appear only once someone reconfigured the bridge.
  const { domain, step } = useMemo<{ domain: [number, number]; step: number }>(() => {
    const plain = (d: [number, number]) => ({ domain: d, step: niceStep(d[1] - d[0]) });
    if (data.length === 0) return plain(series === 'normal' ? [-1, 18] : [-1, 1]);

    // The band, not the mean, bounds the contact trace: an excursion that
    // leaves the plot is the one the scale exists to show.
    let low = Infinity;
    let high = -Infinity;
    for (const d of data) {
      if (series === 'normal') {
        low = Math.min(low, d.fcLo, d.fcHi);
        high = Math.max(high, d.fcLo, d.fcHi);
      } else {
        low = Math.min(low, d.fx, d.fy, d.fz);
        high = Math.max(high, d.fx, d.fy, d.fz);
      }
    }

    if (scale === 'full') {
      if (series === 'normal') {
        return plain([-1, Math.ceil(Math.max(config.maxForceN * 1.2, high))]);
      }
      const peak = Math.max(1, Math.abs(low), Math.abs(high));
      return plain([-Math.ceil(peak), Math.ceil(peak)]);
    }

    // A little headroom so the trace does not ride the frame. The floor is
    // held at zero when the signal never went negative: snapping outward on a
    // wide range would otherwise open the axis down to −5 N, and a quarter of
    // the plate would be given to a region the probe cannot reach.
    const pad = Math.max((high - low) * 0.12, MIN_SPAN_N * 0.12);
    const paddedLow = low >= 0 ? Math.max(0, low - pad) : low - pad;
    return fitRange(paddedLow, high + pad, MIN_SPAN_N);
  }, [data, scale, series]);

  const span = domain[1] - domain[0];
  const decimals = decimalsFor(step);
  const ticks = useMemo(() => ticksFor(domain, step), [domain, step]);
  const inView = (v: number) => v > domain[0] && v < domain[1];
  // The hold band is the reference that matters at a fitted scale: the
  // question stops being "how close to the limit" and becomes "am I holding".
  const holdLow = config.targetForceN - config.targetBandN;
  const holdHigh = config.targetForceN + config.targetBandN;
  // Shade only when an edge of the band is on the plot. A band wider than the
  // view would tint every pixel the same, which carries no information and
  // just dulls the trace — that case is said in words in the key instead.
  const holdEdgeInView = series === 'normal' && (inView(holdLow) || inView(holdHigh));
  const withinHold =
    series === 'normal' && holdLow <= domain[0] && holdHigh >= domain[1];
  const offScale =
    series === 'normal' && !inView(config.maxForceN) && !inView(config.warnForceN);

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Force trend</span>
        <div className={styles.tabs} role="tablist">
          <Tab active={series === 'normal'} onClick={() => setSeries('normal')} label="Contact" />
          <Tab
            active={series === 'components'}
            onClick={() => setSeries('components')}
            label="Components"
          />
          <button
            type="button"
            className={styles.tab}
            onClick={() => setScale(scale === 'fit' ? 'full' : 'fit')}
            title={
              scale === 'fit'
                ? 'Axis is fitted to the signal. Switch to the full force range.'
                : 'Axis spans the full force range. Switch to a fitted axis.'
            }
          >
            {scale === 'fit' ? 'FIT' : 'FULL'}
          </button>
          {/* The span is stated because a fitted axis is only honest if the
              operator can see how much it magnified. Without it a 0.05 N
              wobble and a 5 N ramp draw the same picture. */}
          <span className="plate__aside">
            {span.toFixed(span >= 2 ? 1 : 2)} N{banded ? ` · ${waveformHz} Hz` : ''} · 20 s
          </span>
        </div>
      </div>

      <div className={`plate__body ${styles.body}`}>
        {data.length < 2 ? (
          <p className={styles.empty}>AWAITING FORCE SAMPLES</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 42, bottom: 2, left: -8 }}>
              <CartesianGrid stroke={rule} strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="age"
                type="number"
                domain={[-20, 0]}
                ticks={[-20, -15, -10, -5, 0]}
                tickFormatter={(v: number) => (v === 0 ? 'now' : `${-v}s`)}
                stroke="#4a4a4a"
                tick={AXIS_STYLE}
                tickLine={false}
                axisLine={{ stroke: '#111111' }}
              />
              <YAxis
                domain={domain}
                ticks={ticks}
                width={52}
                stroke="#4a4a4a"
                tick={AXIS_STYLE}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: number) => v.toFixed(decimals)}
                label={{ value: 'N', position: 'insideTopLeft', offset: -2, fill: '#4a4a4a', fontSize: 9 }}
              />
              <Tooltip
                isAnimationActive={false}
                cursor={{ stroke: '#111111', strokeWidth: 1 }}
                contentStyle={{
                  background: '#ffffff',
                  border: `1px solid ${accent}`,
                  fontSize: 11,
                  fontFamily: 'Arial, Helvetica, sans-serif',
                  padding: '4px 7px',
                }}
                labelStyle={{ color: '#111111' }}
                labelFormatter={(v) => `${Math.abs(Number(v)).toFixed(1)} s ago`}
                formatter={(value: number | [number, number], name: string) =>
                  Array.isArray(value)
                    ? [`${value[0].toFixed(3)} … ${value[1].toFixed(3)} N`, name]
                    : [`${value.toFixed(3)} N`, name]
                }
                itemSorter={(item) => -Number(item.value)}
              />

              {series === 'normal' ? (
                <>
                  {/* The hold band, shaded. At a fitted scale this is the
                      reference the operator is actually working against —
                      the regulator's own target ± deadband. */}
                  {holdEdgeInView ? (
                    <ReferenceArea
                      y1={holdLow}
                      y2={holdHigh}
                      fill={accent}
                      fillOpacity={0.07}
                      stroke="none"
                      // `hidden` drops the whole area the moment it reaches
                      // past the axis, which is exactly when half of it is
                      // still on the plot and worth drawing. Clip instead.
                      ifOverflow="visible"
                    />
                  ) : null}
                  {inView(config.targetForceN) ? (
                    <ReferenceLine
                      y={config.targetForceN}
                      stroke={accent}
                      strokeDasharray="4 4"
                      strokeWidth={1}
                      label={{ value: 'Hold', position: 'right', fill: accent, fontSize: 9 }}
                    />
                  ) : null}
                  {/* Rules are drawn only when they are on the plot. Recharts
                      would clamp an out-of-range line to the edge, which reads
                      as "the limit is right there" at a scale where it is
                      ten newtons away. */}
                  {inView(config.warnForceN) ? (
                    <ReferenceLine
                      y={config.warnForceN}
                      stroke="#111111"
                      strokeDasharray="5 3"
                      strokeWidth={1}
                      label={{ value: 'Warn', position: 'right', fill: '#111111', fontSize: 9 }}
                    />
                  ) : null}
                  {inView(config.maxForceN) ? (
                    <ReferenceLine
                      y={config.maxForceN}
                      stroke="#111111"
                      strokeDasharray="2 2"
                      strokeWidth={1.6}
                      label={{ value: 'Limit', position: 'right', fill: '#111111', fontSize: 9 }}
                    />
                  ) : null}
                  {/* The excursion inside each bin, drawn under the mean.
                      Folding a window to its average is only honest if what
                      the average removed is still on the plot — a 0.4 N ripple
                      at 40 Hz is invisible in the mean and is exactly the
                      thing that says the probe is chattering rather than
                      resting. */}
                  {banded ? (
                    <Area
                      type="linear"
                      dataKey="band"
                      name="min–max"
                      stroke="none"
                      fill={accent}
                      fillOpacity={0.2}
                      activeDot={false}
                      isAnimationActive={false}
                    />
                  ) : null}
                  <Line
                    type="linear"
                    dataKey="fc"
                    name="‖F‖"
                    stroke={accent}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </>
              ) : (
                <>
                  <ReferenceLine y={0} stroke="#111111" strokeWidth={1} />
                  <Line type="linear" dataKey="fx" name="Fx" stroke="#111111" strokeWidth={1.1}
                        dot={false} isAnimationActive={false} />
                  <Line type="linear" dataKey="fy" name="Fy" stroke="#4a4a4a" strokeWidth={1.1}
                        dot={false} isAnimationActive={false} />
                  <Line type="linear" dataKey="fz" name="Fz" stroke={accent} strokeWidth={1.5}
                        dot={false} isAnimationActive={false} />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {series === 'components' ? (
        <div className={styles.key}>
          <KeyItem stroke="solid-black" label="Fx" />
          <KeyItem stroke="solid-grey" label="Fy" />
          <KeyItem stroke="solid-navy" label="Fz" />
        </div>
      ) : null}

      {/* Only the normal trace carries a band — the components view is a
          diagnostic for which axis takes the load, and three shaded envelopes
          would obscure the comparison it exists to make. */}
      {series === 'normal' ? (
        <div className={styles.key}>
          {banded ? (
            <>
              <KeyItem
                stroke="solid-navy"
                label={`‖F‖, ${Math.round(1000 / waveformHz)} ms mean`}
              />
              <span className={styles.keyItem}>
                <span className={`${styles.keyLine} ${styles.bandSwatch}`} />
                min–max within each bin
              </span>
            </>
          ) : null}
          {withinHold ? (
            <span className={styles.keyItem}>
              whole view inside the hold band {config.targetForceN.toFixed(1)} ±
              {config.targetBandN.toFixed(1)} N
            </span>
          ) : null}
          {offScale ? (
            <span className={styles.keyItem}>
              Warn {config.warnForceN.toFixed(0)} N · Limit {config.maxForceN.toFixed(0)} N — off
              scale, see the gauge
            </span>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

/**
 * Memoised on its own props.
 *
 * The console has eighteen store subscriptions in one component, and telemetry
 * frames arrive at 30 Hz — so the whole tree re-rendered every frame, and this
 * plot rebuilt two SVG paths each time even though its buffer only changes when
 * the chart flushes at 10 Hz. Measured, that was the difference between 6 fps
 * and something usable on the monitoring view.
 *
 * Both props are stable between flushes: `history` is replaced by reference
 * only when the store hands over a new buffer, and `waveformHz` is a number.
 */
export const ForceTrend = memo(ForceTrendView);

function Tab({ active, onClick, label }: { active: boolean; onClick(): void; label: string }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`${styles.tab} ${active ? styles.tabActive : ''}`}
    >
      {label}
    </button>
  );
}

function KeyItem({ stroke, label }: { stroke: string; label: string }) {
  return (
    <span className={styles.keyItem}>
      <span className={`${styles.keyLine} ${styles[stroke as keyof typeof styles]}`} />
      {label}
    </span>
  );
}
