import { useMemo } from 'react';
import { config } from '../telemetry/config';
import { severityOfForce } from '../store/selectors';
import styles from './ForceGauge.module.css';

interface Props {
  normalForceN: number;
  peakN: number;
  available: boolean;
}

/**
 * Normal-force gauge with the two limits drawn in place.
 *
 * A linear bar rather than a dial: the question the operator asks is "how much
 * margin is left before 7 N", and distance-to-a-mark answers that faster than
 * an angle does. The warn and limit ticks are drawn on the scale itself so no
 * mental arithmetic is needed against a legend.
 *
 * The scale runs slightly past the hard limit so that exceeding it is visible
 * as overshoot rather than as a bar that has simply stopped.
 */
export function ForceGauge({ normalForceN, peakN, available }: Props) {
  const span = config.maxForceN * 1.35;
  const magnitude = Math.abs(normalForceN);
  const severity = severityOfForce(normalForceN);

  const marks = useMemo(
    () => [
      { at: config.contactEnterN, label: 'contact', kind: 'contact' as const },
      { at: config.warnForceN, label: 'warn', kind: 'warn' as const },
      { at: config.maxForceN, label: 'limit', kind: 'limit' as const },
    ],
    [],
  );

  const pct = (value: number) => `${Math.min(100, (value / span) * 100)}%`;

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Normal force · F&#8345;</span>
        <span className={`dot dot--${available ? severity : 'unknown'}`} />
      </div>

      <div className={`panel__body ${styles.body}`}>
        <div className={styles.readout}>
          <span className={`num ${styles.value} ${styles[`value--${available ? severity : 'unknown'}`]}`}>
            {available ? magnitude.toFixed(2) : '—.——'}
          </span>
          <span className={styles.unit}>N</span>
          <div className={styles.peak}>
            <span className="label">Peak</span>
            <span className="num">{available ? peakN.toFixed(2) : '—'}</span>
          </div>
        </div>

        <div className={styles.scale}>
          <div className={styles.track}>
            {/* Zones are drawn under the fill so the fill colour is not fighting
                a background stripe for the same pixels. */}
            <span
              className={styles.zoneWarn}
              style={{ left: pct(config.warnForceN), right: `${100 - (config.maxForceN / span) * 100}%` }}
            />
            <span
              className={styles.zoneLimit}
              style={{ left: pct(config.maxForceN) }}
            />
            <span
              className={`${styles.fill} ${styles[`fill--${severity}`]}`}
              style={{ width: available ? pct(magnitude) : '0%' }}
            />
            {available && peakN > 0 ? (
              <span className={styles.peakMark} style={{ left: pct(peakN) }} />
            ) : null}
            {marks.map((mark) => (
              <span
                key={mark.label}
                className={`${styles.mark} ${styles[`mark--${mark.kind}`]}`}
                style={{ left: pct(mark.at) }}
              />
            ))}
          </div>

          <div className={styles.axis}>
            {marks.map((mark) => (
              <span key={mark.label} className={styles.axisLabel} style={{ left: pct(mark.at) }}>
                <span className={styles.axisValue}>{mark.at.toFixed(1)}</span>
                <span className={styles.axisName}>{mark.label}</span>
              </span>
            ))}
          </div>
        </div>

        <p className={styles.note}>
          F&#8345; = {config.normalForceSign > 0 ? '+' : '−'}F&#7827; in the probe frame; positive is
          compression. Sign convention is provisional and awaits bench verification.
        </p>
      </div>
    </section>
  );
}
