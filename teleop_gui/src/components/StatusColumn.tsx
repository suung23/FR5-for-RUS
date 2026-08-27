import { config } from '../telemetry/config';
import { GATE_CONTENT, type GateTopic } from '../telemetry/guidanceGate';
import { JOINT_NAMES, marginToLimit } from '../telemetry/fr5Model';
import type { ContactSnapshot } from '../telemetry/contactState';
import type { RobotTelemetry, WrenchSample, WrenchSource } from '../telemetry/types';
import { SAFETY_STATE_LABEL } from '../store/selectors';
import styles from './StatusColumn.module.css';

const RAD_TO_DEG = 180 / Math.PI;
const NEAR_LIMIT_DEG = 10;

interface Props {
  telemetry: RobotTelemetry;
  wrench: WrenchSample | null;
  contact: ContactSnapshot;
  peakN: number;
  available: boolean;
  /** Last guidance the operator acknowledged, or null before the first. */
  acknowledged: { topic: GateTopic; at: number } | null;
  frameCount: number;
}

/**
 * Right-hand numeric column.
 *
 * Read top to bottom: where the tool is, what the joints are doing, what the
 * sensor reads, what the operator should do. Everything is a labelled field
 * with a right-aligned tabular value, so the column can be scanned down the
 * numbers without reading the labels again.
 */
export function StatusColumn({
  telemetry,
  wrench,
  contact,
  peakN,
  available,
  acknowledged,
  frameCount,
}: Props) {
  return (
    <aside className={styles.column}>
      {/* Stage, force and guidance are pinned. Everything between them scrolls.
          Which plates are visible must not depend on whether the source
          happens to publish a TCP pose today — the readings an operator acts
          on cannot scroll off the screen because an optional field arrived. */}
      <div className={styles.pinned}>
        <StagePlate contact={contact} telemetry={telemetry} available={available} />
        <ForceScale
          contact={contact}
          peakN={peakN}
          available={available && wrench !== null}
          sampleCount={frameCount}
        />
        {/* The six raw channels are pinned alongside the derived normal force.
            When the sensor's axis assignment is still provisional, the raw
            reading is what an operator checks against — burying it below a
            scroll makes the derived number the only one in view. */}
        <WrenchPlate wrench={available ? wrench : null} />
      </div>

      <div className={styles.scroll}>
        <TcpPlate telemetry={telemetry} available={available} />
        <JointPlate telemetry={telemetry} available={available} />
      </div>

      <div className={styles.pinned}>
        <AcknowledgedLine acknowledged={acknowledged} />
      </div>
    </aside>
  );
}

/**
 * Stage and safety as plain named fields.
 *
 * No badge, no tile. The stage is a word in a value slot like every other
 * reading, because it is one — and treating it as a graphic invites the
 * operator to read the shape instead of the text.
 */
function StagePlate({
  contact,
  telemetry,
  available,
}: {
  contact: ContactSnapshot;
  telemetry: RobotTelemetry;
  available: boolean;
}) {
  const safety = telemetry.safetyState;
  const probingMode = telemetry.probingMode;
  const alarm = safety === 'protective_stop' || safety === 'emergency_stop';
  const warn = safety === 'warning';

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Operating stage</span>
      </div>
      <div className="plate__body">
        <div className={`field ${contact.phase === 'contact' ? 'field--warn' : ''}`}>
          <span className="field__label">Stage</span>
          <span className="field__value">
            <span className={contact.phase === 'contact' ? 'tag tag--strong' : 'tag tag--off'}>
              {available ? (contact.phase === 'contact' ? 'CONTACT' : 'APPROACH') : 'UNKNOWN'}
            </span>
          </span>
        </div>
        <div className={`field ${alarm ? 'field--alarm' : warn ? 'field--warn' : ''}`}>
          <span className="field__label">Safety</span>
          <span className="field__value num">
            {available && safety ? SAFETY_STATE_LABEL[safety].toUpperCase() : '—'}
            {alarm ? <span className={styles.inlineTag}>ALARM</span> : null}
            {warn ? <span className={styles.inlineTagLight}>WARN</span> : null}
          </span>
        </div>
        <div className={`field ${contact.hasContacted ? 'field--warn' : ''}`}>
          <span className="field__label">Contact latch</span>
          <span className="field__value num">
            {contact.hasContacted ? 'ENGAGED' : 'NOT ENGAGED'}
          </span>
        </div>
        {/* What the robot is actually clamping to, as reported by the control
            stack — not the console's own guess. When these two disagree the
            operator needs to see it, so the declared mode is its own field. */}
        <div className={`field ${probingMode === 'contact_probing' ? 'field--warn' : ''}`}>
          <span className="field__label">Velocity limits</span>
          <span className="field__value num">
            {!available || !probingMode
              ? '—'
              : probingMode === 'contact_probing'
                ? 'CONTACT PROBING'
                : 'APPROACH'}
          </span>
        </div>
        <p className={styles.note}>
          Stage comes from the force sensor on its own USB link, not from the robot
          controller. The two can disagree.
        </p>
      </div>
    </section>
  );
}

/**
 * Linear force scale.
 *
 * A ruled track with the measured value marked in navy, warn as a grey rule and
 * the limit as a heavy black rule. The numeric reading above it is the primary
 * source; the track exists to show margin, which a number alone does not.
 */
function ForceScale({
  contact,
  peakN,
  available,
  sampleCount,
}: {
  contact: ContactSnapshot;
  peakN: number;
  available: boolean;
  sampleCount: number;
}) {
  const span = config.maxForceN * 1.3;
  const magnitude = Math.abs(contact.normalForceN);
  const pct = (v: number) => `${Math.min(100, Math.max(0, (v / span) * 100))}%`;
  const overLimit = magnitude >= config.maxForceN;
  const overWarn = magnitude >= config.warnForceN;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Normal force</span>
        <span className="plate__aside">Fn = −Fz · {sampleCount} buffered</span>
      </div>
      <div className={`plate__body ${styles.forceBody}`}>
        <div className={styles.forceRead}>
          <span className={`num ${styles.forceValue} ${overLimit ? styles.forceValueAlarm : ''}`}>
            {available ? magnitude.toFixed(2) : '—.——'}
          </span>
          <span className={styles.forceUnit}>N</span>
          {overLimit ? <span className="tag tag--strong">OVER LIMIT</span> : null}
          {!overLimit && overWarn ? <span className="tag">WARN</span> : null}
          {!overLimit && !overWarn && magnitude >= config.contactProbingN ? (
            <span className="tag tag--navy">PROBING</span>
          ) : null}
          <span className={styles.forcePeak}>
            <span className={styles.peakLabel}>PEAK</span>
            <span className="num">{available ? peakN.toFixed(2) : '—'}</span>
          </span>
        </div>

        <div className={styles.scale}>
          {/* White gauge, black scale. The reading is a deep-green marker, not a
              filled bar — a filled bar reads as "how much of the budget is
              used", and what the operator needs is where the value sits
              relative to the target band and the limit. */}
          <div className={styles.track}>
            <span
              className={styles.safeBand}
              style={{
                left: pct(config.targetForceN - config.targetBandN),
                width: pct(2 * config.targetBandN),
              }}
            />
            <span className={styles.targetMark} style={{ left: pct(config.targetForceN) }} />
            <span className={styles.probingMark} style={{ left: pct(config.contactProbingN) }} />
            <span className={styles.warnMark} style={{ left: pct(config.warnForceN) }} />
            <span className={styles.limitMark} style={{ left: pct(config.maxForceN) }} />
            {available && peakN > 0 ? (
              <span className={styles.peakMark} style={{ left: pct(peakN) }} />
            ) : null}
            {available ? (
              <span className={styles.measuredMark} style={{ left: pct(magnitude) }} />
            ) : null}
          </div>
          <div className={styles.scaleTicks}>
            <span style={{ left: '0%' }}>0</span>
            <span style={{ left: pct(config.targetForceN) }}>
              {config.targetForceN.toFixed(0)}
            </span>
            <span style={{ left: pct(config.contactProbingN) }}>
              {config.contactProbingN.toFixed(0)}
            </span>
            <span style={{ left: pct(config.warnForceN) }}>{config.warnForceN.toFixed(0)}</span>
            <span style={{ left: pct(config.maxForceN) }}>{config.maxForceN.toFixed(0)}</span>
          </div>
          <div className={styles.scaleKey}>
            <span>
              <i className={styles.keyMeasured} /> Measured
            </span>
            <span>
              <i className={styles.keyTarget} /> Hold band {config.targetForceN.toFixed(1)} ±
              {config.targetBandN.toFixed(1)} N
            </span>
            <span>
              <i className={styles.keyProbing} /> Probing {config.contactProbingN.toFixed(1)} N
            </span>
            <span>
              <i className={styles.keyWarn} /> Warn {config.warnForceN.toFixed(1)} N
            </span>
            <span>
              <i className={styles.keyLimit} /> Limit {config.maxForceN.toFixed(1)} N
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * TCP pose.
 *
 * Collapses to a single explanatory line when the source does not publish one,
 * rather than showing four dashes. Four empty slots take the height of real
 * data and give the operator nothing to read.
 */
function TcpPlate({ telemetry, available }: { telemetry: RobotTelemetry; available: boolean }) {
  const pose = available ? telemetry.tcpPose : undefined;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">TCP pose</span>
        <span className="plate__aside">base frame · mm</span>
      </div>
      <div className="plate__body">
        {pose ? (
          <>
            {(['X', 'Y', 'Z'] as const).map((axis, i) => (
              <div className="field" key={axis}>
                <span className="field__label">{axis}</span>
                <span className="field__value num">
                  {(pose.position[i] * 1000).toFixed(1)}
                </span>
              </div>
            ))}
            <div className="field">
              <span className="field__label">Quaternion</span>
              <span className="field__value num">
                {pose.quaternion.map((q) => q.toFixed(3)).join(' ')}
              </span>
            </div>
          </>
        ) : (
          <div className="field">
            <span className="field__label">Not published</span>
            <span className="field__value num">FK FROM JOINTS</span>
          </div>
        )}
      </div>
    </section>
  );
}

function JointPlate({ telemetry, available }: { telemetry: RobotTelemetry; available: boolean }) {
  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Joint state</span>
        <span className="plate__aside">deg · deg/s · margin to stop</span>
      </div>
      <div className="plate__body">
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Axis</th>
              <th scope="col">Position</th>
              <th scope="col">Rate</th>
              <th scope="col">Margin</th>
            </tr>
          </thead>
          <tbody>
            {JOINT_NAMES.map((name, i) => {
              const q = available ? telemetry.jointPositions?.[i] : undefined;
              const v = available ? telemetry.jointVelocities?.[i] : undefined;
              const known = typeof q === 'number';
              const marginDeg = known ? marginToLimit(i, q) * RAD_TO_DEG : undefined;
              const near = marginDeg !== undefined && marginDeg < NEAR_LIMIT_DEG;
              return (
                <tr key={name} className={near ? styles.rowNear : undefined}>
                  <th scope="row">{name}</th>
                  <td className="num">{known ? (q * RAD_TO_DEG).toFixed(2) : '—'}</td>
                  <td className="num">{typeof v === 'number' ? (v * RAD_TO_DEG).toFixed(1) : '—'}</td>
                  <td className="num">
                    {marginDeg !== undefined ? marginDeg.toFixed(1) : '—'}
                    {near ? <span className={styles.nearTag}>LIM</span> : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const WRENCH_SOURCE_LABEL: Record<WrenchSource, string> = {
  px6d_serial: 'PX6D · USB DIRECT',
  controller: 'VIA CONTROLLER',
  simulation: 'SIMULATED',
  none: 'NO SOURCE',
};

/**
 * Six-axis force and torque, with its origin stated.
 *
 * The header used to read "PX6D · raw" whatever was feeding it. On this cell
 * that was usually wrong: the PX6D is the USB variant, so it never reaches the
 * robot controller, and a wrench arriving on the controller topic is somebody
 * else's sensor or zeros. The plate now names the actual source, and when the
 * sensor is read directly it also carries the link quality — so "the numbers
 * look wrong" and "the cable is bad" can be told apart without leaving the
 * console.
 */
function WrenchPlate({ wrench }: { wrench: WrenchSample | null }) {
  const rows: [string, number | undefined, string][] = [
    ['Fx', wrench?.force[0], 'N'],
    ['Fy', wrench?.force[1], 'N'],
    ['Fz', wrench?.force[2], 'N'],
    ['Mx', wrench?.torque[0], 'N·m'],
    ['My', wrench?.torque[1], 'N·m'],
    ['Mz', wrench?.torque[2], 'N·m'],
  ];

  const source = wrench?.source;
  const direct = source === 'px6d_serial';
  const badLink = direct && (wrench?.crcErrors ?? 0) > 0;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Force / torque</span>
        <span className="plate__aside">
          {source ? WRENCH_SOURCE_LABEL[source] : 'SOURCE UNKNOWN'}
        </span>
      </div>
      <div className="plate__body">
        <table className={styles.table}>
          <tbody>
            {rows.map(([label, value, unit]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td className={`num ${styles.wide}`}>
                  {value === undefined ? '—' : value.toFixed(unit === 'N' ? 3 : 4)}
                </td>
                <td className={styles.unit}>{unit}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {direct ? (
          <div className={styles.sensorLink}>
            <span className={styles.sensorItem}>
              <span className={styles.sensorLabel}>RATE</span>
              <span className="num">{(wrench?.sensorHz ?? 0).toFixed(0)} Hz</span>
            </span>
            <span className={`${styles.sensorItem} ${badLink ? styles.sensorBad : ''}`}>
              <span className={styles.sensorLabel}>CRC</span>
              <span className="num">{wrench?.crcErrors ?? 0}</span>
            </span>
            <span className={styles.sensorNote}>
              {badLink ? 'CHECK CABLE' : 'LINK CLEAN'}
            </span>
          </div>
        ) : (
          <p className={styles.tableNote}>
            {source === 'controller'
              ? 'Arriving on the controller wrench topic. Our PX6D is the USB variant and does not reach the controller — start the bridge with bridge.px6d_port to read it directly.'
              : source === 'simulation'
                ? 'Generated by the built-in simulator. No sensor is connected.'
                : 'No force source is publishing.'}
          </p>
        )}
      </div>
    </section>
  );
}

/**
 * Acknowledgement receipt.
 *
 * Replaces the old standing guidance panel. Standing text in a corner is read
 * once and then becomes furniture; the checklist now arrives as a gate the
 * operator has to dismiss. What remains here is only the record that they did —
 * which mode, and when.
 *
 * Deliberately not interactive: it is a receipt, not a way back into the text.
 */
function AcknowledgedLine({
  acknowledged,
}: {
  acknowledged: { topic: GateTopic; at: number } | null;
}) {
  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Guidance</span>
      </div>
      <div className="plate__body">
        <div className="field">
          <span className="field__label">Acknowledged</span>
          <span className="field__value num">
            {acknowledged
              ? `${GATE_CONTENT[acknowledged.topic].title} · ${formatClock(acknowledged.at)}`
              : '—'}
          </span>
        </div>
      </div>
    </section>
  );
}

function formatClock(at: number): string {
  return new Date(at).toLocaleTimeString('en-GB', { hour12: false });
}
