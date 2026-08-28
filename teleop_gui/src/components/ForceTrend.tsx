import { useMemo, useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { config } from '../telemetry/config';
import type { ForcePoint } from '../store/telemetryStore';
import styles from './ForceTrend.module.css';

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

const AXIS_STYLE = { fontSize: 9.5, fill: '#4a4a4a', fontFamily: 'Arial, Helvetica, sans-serif' };

/**
 * Rolling force trend.
 *
 * Two series, chosen by a tab rather than crammed together. `NORMAL` answers
 * the operating question — how much margin is left to the limit — with the
 * warn and limit rules drawn on the plot. `COMPONENTS` answers the diagnostic
 * one about which axis carries the load, which still matters while the
 * sensor's axis assignment is provisional.
 *
 * Traces are black; only the measured normal force is navy, so the line the
 * operator is actually reading is the one that differs.
 */
export function ForceTrend({ history, waveformHz }: Props) {
  const [series, setSeries] = useState<Series>('normal');

  const data = useMemo(() => {
    if (history.length === 0) return [];
    const now = history[history.length - 1].t;
    // `band` is the pair recharts draws a range area from. Built here rather
    // than read as two keys because a range series takes one key holding both
    // edges, and building it in the view keeps the store's point a plain record.
    return history.map((p) => ({
      ...p,
      age: (p.t - now) / 1000,
      band: [p.fnLo, p.fnHi] as [number, number],
    }));
  }, [history]);

  const banded = waveformHz !== null;

  // Scanned in a loop rather than spread into `Math.max`. At the bridge's
  // default the buffer is a couple of thousand points, and a waveform rate set
  // higher would push a spread past the engine's argument limit — a crash that
  // would appear only once someone reconfigured the bridge.
  const domain = useMemo<[number, number]>(() => {
    if (series === 'normal') {
      // The band, not the mean, sets the top: an excursion that leaves the
      // plot is the one reading the scale exists for.
      let peak = config.maxForceN * 1.2;
      for (const d of data) peak = Math.max(peak, Math.abs(d.fnHi), Math.abs(d.fnLo));
      return [-1, Math.ceil(peak)];
    }
    let peak = 1;
    for (const d of data) {
      peak = Math.max(peak, Math.abs(d.fx), Math.abs(d.fy), Math.abs(d.fz));
    }
    return [-Math.ceil(peak), Math.ceil(peak)];
  }, [data, series]);

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Force trend</span>
        <div className={styles.tabs} role="tablist">
          <Tab active={series === 'normal'} onClick={() => setSeries('normal')} label="Normal" />
          <Tab
            active={series === 'components'}
            onClick={() => setSeries('components')}
            label="Components"
          />
          <span className="plate__aside">
            {banded ? `${waveformHz} Hz · 20 s` : '20 s'}
          </span>
        </div>
      </div>

      <div className={`plate__body ${styles.body}`}>
        {data.length < 2 ? (
          <p className={styles.empty}>AWAITING FORCE SAMPLES</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 42, bottom: 2, left: -8 }}>
              <CartesianGrid stroke="#b9d0be" strokeDasharray="0" vertical={false} />
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
                width={42}
                stroke="#4a4a4a"
                tick={AXIS_STYLE}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v: number) => v.toFixed(0)}
                label={{ value: 'N', position: 'insideTopLeft', offset: -2, fill: '#4a4a4a', fontSize: 9 }}
              />
              <Tooltip
                isAnimationActive={false}
                cursor={{ stroke: '#111111', strokeWidth: 1 }}
                contentStyle={{
                  background: '#ffffff',
                  border: '1px solid #17683a',
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
              />

              {series === 'normal' ? (
                <>
                  <ReferenceLine
                    y={config.warnForceN}
                    stroke="#111111"
                    strokeDasharray="5 3"
                    strokeWidth={1}
                    label={{ value: 'Warn', position: 'right', fill: '#111111', fontSize: 9 }}
                  />
                  <ReferenceLine
                    y={config.maxForceN}
                    stroke="#111111"
                    strokeDasharray="2 2"
                    strokeWidth={1.6}
                    label={{ value: 'Limit', position: 'right', fill: '#111111', fontSize: 9 }}
                  />
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
                      fill="#17683a"
                      fillOpacity={0.2}
                      activeDot={false}
                      isAnimationActive={false}
                    />
                  ) : null}
                  <Line
                    type="linear"
                    dataKey="fn"
                    name="Fn"
                    stroke="#17683a"
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
                  <Line type="linear" dataKey="fz" name="Fz" stroke="#17683a" strokeWidth={1.5}
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
      {series === 'normal' && banded ? (
        <div className={styles.key}>
          <KeyItem stroke="solid-navy" label={`Fn, ${Math.round(1000 / waveformHz)} ms mean`} />
          <span className={styles.keyItem}>
            <span className={`${styles.keyLine} ${styles.bandSwatch}`} />
            min–max within each bin
          </span>
        </div>
      ) : null}
    </section>
  );
}

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
