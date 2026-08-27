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
import styles from './ForceTrend.module.css';

interface Props {
  history: ForcePoint[];
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
export function ForceTrend({ history }: Props) {
  const [series, setSeries] = useState<Series>('normal');

  const data = useMemo(() => {
    if (history.length === 0) return [];
    const now = history[history.length - 1].t;
    return history.map((p) => ({ ...p, age: (p.t - now) / 1000 }));
  }, [history]);

  const domain = useMemo<[number, number]>(() => {
    if (series === 'normal') {
      const peak = Math.max(config.maxForceN * 1.2, ...data.map((d) => Math.abs(d.fn)));
      return [-1, Math.ceil(peak)];
    }
    const peak = Math.max(1, ...data.flatMap((d) => [Math.abs(d.fx), Math.abs(d.fy), Math.abs(d.fz)]));
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
          <span className="plate__aside">20 s</span>
        </div>
      </div>

      <div className={`plate__body ${styles.body}`}>
        {data.length < 2 ? (
          <p className={styles.empty}>AWAITING FORCE SAMPLES</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 42, bottom: 2, left: -8 }}>
              <CartesianGrid stroke="#d5ded7" strokeDasharray="0" vertical={false} />
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
                  border: '1px solid #1e6b45',
                  fontSize: 11,
                  fontFamily: 'Arial, Helvetica, sans-serif',
                  padding: '4px 7px',
                }}
                labelStyle={{ color: '#111111' }}
                labelFormatter={(v) => `${Math.abs(Number(v)).toFixed(1)} s ago`}
                formatter={(value: number, name: string) => [`${value.toFixed(3)} N`, name]}
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
                  <Line
                    type="linear"
                    dataKey="fn"
                    name="Fn"
                    stroke="#1e6b45"
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
                  <Line type="linear" dataKey="fz" name="Fz" stroke="#1e6b45" strokeWidth={1.5}
                        dot={false} isAnimationActive={false} />
                </>
              )}
            </LineChart>
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
