import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { config } from '../telemetry/config';
import type { ForcePoint } from '../store/telemetryStore';
import styles from './ForceChart.module.css';

interface Props {
  history: ForcePoint[];
  wrench: [number, number, number] | null;
  torque: [number, number, number] | null;
}

type Mode = 'normal' | 'components';

const TRACES = [
  { key: 'fx' as const, label: 'Fx', colorVar: '--trace-x' },
  { key: 'fy' as const, label: 'Fy', colorVar: '--trace-y' },
  { key: 'fz' as const, label: 'Fz', colorVar: '--trace-z' },
];

/**
 * Rolling force plot.
 *
 * Two views rather than one crowded one. `normal` answers the operating
 * question — how close is F_n to the limit, and how is it trending — with the
 * warn and limit lines drawn in. `components` answers the diagnostic question
 * about which axis is loaded, which matters while the sensor's axis assignment
 * is still provisional.
 *
 * The x axis is seconds *before now* rather than wall-clock time. On a scrolling
 * window relative age is what is actually being read, and it avoids an axis
 * whose labels change every tick.
 */
export function ForceChart({ history, wrench, torque }: Props) {
  const [mode, setMode] = useState<Mode>('normal');

  const data = useMemo(() => {
    if (history.length === 0) return [];
    const now = history[history.length - 1].t;
    return history.map((p) => ({ ...p, age: (p.t - now) / 1000 }));
  }, [history]);

  const domain = useMemo<[number, number]>(() => {
    if (mode === 'normal') {
      const peak = Math.max(config.maxForceN * 1.15, ...data.map((d) => Math.abs(d.fn)));
      return [Math.min(-0.5, -peak * 0.1), peak];
    }
    const peak = Math.max(1, ...data.flatMap((d) => [Math.abs(d.fx), Math.abs(d.fy), Math.abs(d.fz)]));
    return [-peak * 1.1, peak * 1.1];
  }, [data, mode]);

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Force history · {history.length > 0 ? '20 s window' : 'waiting'}</span>
        <div className={styles.toggle} role="group" aria-label="Chart series">
          <button
            type="button"
            className={mode === 'normal' ? styles.active : undefined}
            onClick={() => setMode('normal')}
            aria-pressed={mode === 'normal'}
          >
            Normal
          </button>
          <button
            type="button"
            className={mode === 'components' ? styles.active : undefined}
            onClick={() => setMode('components')}
            aria-pressed={mode === 'components'}
          >
            Components
          </button>
        </div>
      </div>

      <div className={`panel__body panel__body--flush ${styles.body}`}>
        <div className={styles.chart}>
          {data.length < 2 ? (
            <p className={styles.empty}>Waiting for force samples…</p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 12, right: 14, bottom: 4, left: -6 }}>
                <CartesianGrid stroke="var(--hairline)" strokeDasharray="2 4" vertical={false} />
                <XAxis
                  dataKey="age"
                  type="number"
                  domain={[-20, 0]}
                  ticks={[-20, -15, -10, -5, 0]}
                  tickFormatter={(v: number) => (v === 0 ? 'now' : `${v}s`)}
                  stroke="var(--ink-faint)"
                  tick={{ fontSize: 10, fill: 'var(--ink-faint)' }}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--hairline)' }}
                />
                <YAxis
                  domain={domain}
                  width={44}
                  stroke="var(--ink-faint)"
                  tick={{ fontSize: 10, fill: 'var(--ink-faint)', fontFamily: 'var(--font-num)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => v.toFixed(1)}
                  label={{
                    value: 'N',
                    position: 'insideTopLeft',
                    offset: 8,
                    fill: 'var(--ink-faint)',
                    fontSize: 10,
                  }}
                />
                <Tooltip
                  isAnimationActive={false}
                  cursor={{ stroke: 'var(--hairline-strong)', strokeWidth: 1 }}
                  contentStyle={{
                    background: 'var(--surface-3)',
                    border: '1px solid var(--hairline-strong)',
                    borderRadius: 4,
                    fontSize: 11,
                    fontFamily: 'var(--font-num)',
                    padding: '6px 8px',
                  }}
                  labelFormatter={(v) => `${Number(v).toFixed(1)} s`}
                  formatter={(value: number, name: string) => [`${value.toFixed(3)} N`, name]}
                />

                {mode === 'normal' ? (
                  <>
                    <ReferenceLine
                      y={config.warnForceN}
                      stroke="var(--caution)"
                      strokeDasharray="4 3"
                      strokeWidth={1}
                      label={{ value: 'warn', position: 'right', fill: 'var(--caution)', fontSize: 9 }}
                    />
                    <ReferenceLine
                      y={config.maxForceN}
                      stroke="var(--critical)"
                      strokeWidth={1.2}
                      label={{ value: 'limit', position: 'right', fill: 'var(--critical)', fontSize: 9 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="fn"
                      name="Fn"
                      stroke="var(--trace-n)"
                      strokeWidth={1.6}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </>
                ) : (
                  TRACES.map((trace) => (
                    <Line
                      key={trace.key}
                      type="monotone"
                      dataKey={trace.key}
                      name={trace.label}
                      stroke={`var(${trace.colorVar})`}
                      strokeWidth={1.4}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))
                )}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <ChannelStrip wrench={wrench} torque={torque} />
      </div>
    </section>
  );
}

function ChannelStrip({
  wrench,
  torque,
}: {
  wrench: [number, number, number] | null;
  torque: [number, number, number] | null;
}) {
  const cells = [
    { label: 'Fx', value: wrench?.[0], unit: 'N', colorVar: '--trace-x' },
    { label: 'Fy', value: wrench?.[1], unit: 'N', colorVar: '--trace-y' },
    { label: 'Fz', value: wrench?.[2], unit: 'N', colorVar: '--trace-z' },
    { label: 'Mx', value: torque?.[0], unit: 'N·m', colorVar: undefined },
    { label: 'My', value: torque?.[1], unit: 'N·m', colorVar: undefined },
    { label: 'Mz', value: torque?.[2], unit: 'N·m', colorVar: undefined },
  ];

  return (
    <div className={styles.channels}>
      {cells.map((cell) => (
        <div key={cell.label} className={styles.channel}>
          <span
            className={styles.swatch}
            style={{ background: cell.colorVar ? `var(${cell.colorVar})` : 'var(--ink-faint)' }}
          />
          <span className={styles.channelLabel}>{cell.label}</span>
          <span className={`num ${styles.channelValue}`}>
            {cell.value === undefined ? '—' : cell.value.toFixed(cell.unit === 'N' ? 3 : 4)}
          </span>
          <span className={styles.channelUnit}>{cell.unit}</span>
        </div>
      ))}
    </div>
  );
}
