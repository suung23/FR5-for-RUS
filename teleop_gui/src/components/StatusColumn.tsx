import { fixed, signed } from '../lib/format';
import { config } from '../telemetry/config';
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
  /** When the operator acknowledged the briefing, or null before they have. */
  acknowledged: { at: number } | null;
  frameCount: number;
  /** Operator's display zero, or null when the readings are untared. */
  forceZero: { vector: [number, number, number]; mode: 'raw' | 'compensated'; at: number } | null;
  onZeroForce(): boolean;
  onClearForceZero(): void;
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
  forceZero,
  onZeroForce,
  onClearForceZero,
}: Props) {
  // A zero taken on raw channels describes a different pipeline stage than a
  // compensated reading does. Rather than subtract it from a number it was
  // never measured against, the zero is set aside and the plate says so.
  const mode: 'raw' | 'compensated' = wrench?.compensated ? 'compensated' : 'raw';
  const zeroApplies = forceZero !== null && forceZero.mode === mode;
  const zero = zeroApplies ? forceZero : null;

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
          zeroNormalN={zero ? zero.vector[2] : 0}
          zeroed={zero !== null}
        />
        {/* The normal force is one component of the contact force, not all of
            it. A probe dragging sideways can sit inside the normal-force band
            and still be loading the tool — that shear is what reaches the
            moment limit through a 201 mm lever. */}
        <ContactForcePlate
          wrench={available ? wrench : null}
          zero={zero}
          staleZero={forceZero !== null && !zeroApplies}
          onZero={onZeroForce}
          onClear={onClearForceZero}
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
/**
 * Which scale ticks can be labelled without their text colliding.
 *
 * Walks from the top down and keeps a tick only when it is far enough below
 * the last one kept. The limit is always kept — it is the number the operator
 * is judging everything else against.
 */
function tickLabels(fullScaleN: number, values: number[]): number[] {
  const minGap = fullScaleN * 0.12;
  const kept: number[] = [];
  for (const value of [...values].sort((a, b) => b - a)) {
    if (kept.length === 0 || kept[kept.length - 1] - value >= minGap) kept.push(value);
  }
  return kept.sort((a, b) => a - b);
}

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
  zeroNormalN,
  zeroed,
}: {
  contact: ContactSnapshot;
  peakN: number;
  available: boolean;
  sampleCount: number;
  /** Offset subtracted from the reading, in newtons. Zero when untared. */
  zeroNormalN: number;
  zeroed: boolean;
}) {
  const span = config.maxForceN * 1.3;
  // The classifier keeps running on the untared value — contact is a physical
  // judgement, not a display preference — so only what is drawn moves here.
  const magnitude = Math.abs(contact.normalForceN - zeroNormalN);
  const pct = (v: number) => `${Math.min(100, Math.max(0, (v / span) * 100))}%`;
  const overLimit = magnitude >= config.maxForceN;
  const overWarn = magnitude >= config.warnForceN;

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Normal force</span>
        <span className="plate__aside">
          {zeroed ? <span className="tag tag--off">ZEROED</span> : null} Fn = −Fz ·{' '}
          {sampleCount} buffered
        </span>
      </div>
      <div className={`plate__body ${styles.forceBody}`}>
        <div className={styles.forceRead}>
          <span className={`num ${styles.forceValue} ${overLimit ? styles.forceValueAlarm : ''}`}>
            {available ? magnitude.toFixed(1).padStart(5, '\u2007') : '—.—'}
          </span>
          <span className={styles.forceUnit}>N</span>
          {overLimit ? <span className="tag tag--strong">OVER LIMIT</span> : null}
          {!overLimit && overWarn ? <span className="tag">WARN</span> : null}
          {!overLimit && !overWarn && magnitude >= config.contactProbingN ? (
            <span className="tag tag--navy">PROBING</span>
          ) : null}
          <span className={styles.forcePeak}>
            <span className={styles.peakLabel}>PEAK</span>
            <span className="num">{available ? peakN.toFixed(1) : '—'}</span>
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
          {/* Numbered ticks are dropped when they would collide with the one
              after them. Warn and limit sit 1 N apart, and on a 0–15 scale
              their labels ran together into "1415" — a tick that cannot be
              read is worse than no tick, because it still looks like a number.
              Every threshold is named in full in the key below, so nothing is
              lost by leaving the crowded one unlabelled; the line stays. */}
          <div className={styles.scaleTicks}>
            {tickLabels(config.maxForceN, [
              0,
              config.targetForceN,
              config.contactProbingN,
              config.warnForceN,
              config.maxForceN,
            ]).map((value) => (
              <span key={value} style={{ left: pct(value) }}>{value.toFixed(0)}</span>
            ))}
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
/**
 * A number whose printed width never changes.
 *
 * `toFixed` drops the sign on positives, so a value crossing zero loses a
 * character and every column to its right steps sideways. In a plate that
 * updates a hundred times a second that reads as a permanent shimmer, and the
 * eye follows the movement rather than the number. Writing the sign always —
 * `+0.0` as readily as `-0.0` — fixes the width, and the sign is worth seeing
 * on a force reading anyway.
 */
/**
 * A rate rounded to a step, so it stops moving.
 *
 * Sample counts wander by a few either side of their nominal value. Printed
 * exactly, that puts a digit changing every frame next to readings that were
 * deliberately calmed down, and the whole row shimmers. The step is chosen so
 * a genuine change — a link falling to half rate — still shows.
 */
function quantise(value: number | undefined, step: number): string {
  if (value === undefined || !Number.isFinite(value) || value <= 0) return '—';
  return String(Math.round(value / step) * step);
}

/**
 * The contact force, decomposed at the probe tip.
 *
 * The big number above is one component of this vector — the part along the
 * penetration axis. It is the one the force regulator closes on, so it earns
 * the size. But a probe can sit right in the hold band and still be loaded
 * sideways, and nothing above would say so: `Fn` is blind to shear by
 * construction.
 *
 * Shear is not a curiosity here. It runs through a 201 mm lever to the sensor,
 * so 1.49 N of it reaches the 0.3 N·m moment limit — a force an operator would
 * not think twice about if they only watched the normal reading.
 *
 * The off-axis angle is the one number that says "you are not pressing
 * straight in". Two forces can share a total and mean opposite things; the
 * angle separates them without the operator doing the arithmetic.
 *
 * **Where the numbers come from is stated, every frame.** With a calibration
 * loaded these are the compensated contact-point values — payload removed,
 * rotated into the probe frame. Without one they are the raw channels, which
 * still carry the tool's own weight, and the plate says so rather than
 * presenting a total that is partly the probe weighing itself.
 */
function ContactForcePlate({
  wrench,
  zero,
  staleZero,
  onZero,
  onClear,
}: {
  wrench: WrenchSample | null;
  zero: { vector: [number, number, number]; mode: 'raw' | 'compensated'; at: number } | null;
  /** A zero exists but was taken against a different pipeline stage. */
  staleZero: boolean;
  onZero(): boolean;
  onClear(): void;
}) {
  const compensated = wrench?.compensated;
  // `contactProbe` is [Fx, Fy, Fz, Mx, My, Mz] in the probe frame, already at
  // the contact point. Its +z is compression, so no sign flip belongs here —
  // the flip below is only for the raw channels, which are in sensor axes.
  const measured = compensated
    ? (compensated.contactProbe.slice(0, 3) as number[])
    : wrench
      ? [wrench.force[0], wrench.force[1], config.normalForceSign * wrench.force[2]]
      : null;
  // The zero is subtracted as a vector, not as a magnitude. Taking it off the
  // total instead would leave the direction untouched, and the direction is
  // most of what this plate is for.
  const force =
    measured && zero
      ? measured.map((v, i) => v - zero.vector[i])
      : measured;

  const normal = force ? force[2] : null;
  const shear = force ? Math.hypot(force[0], force[1]) : null;
  const total = force ? Math.hypot(force[0], force[1], force[2]) : null;
  // Measured from the penetration axis. Undefined at zero load, where the
  // direction of a force that is not there would be noise dressed as an angle.
  const offAxis =
    shear !== null && normal !== null && Math.hypot(shear, normal) > 0.2
      ? (Math.atan2(shear, normal) * 180) / Math.PI
      : null;

  // Total, shear and the angle are magnitudes and cannot go negative, so they
  // keep their width without a sign. Normal can — it goes negative the moment
  // the probe is pulled rather than pressed, and after a zero it sits either
  // side of nothing — so it always carries one.
  const rows: [string, string, string][] = [
    ['Total', fixed(total, 1), 'N'],
    ['Normal', signed(normal, 1), 'N'],
    ['Shear', fixed(shear, 1), 'N'],
    ['Off-axis', fixed(offAxis, 0), '°'],
  ];

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Contact force</span>
        <span className="plate__aside">
          {zero ? <span className="tag tag--off">ZEROED</span> : null}{' '}
          {compensated ? (
            'probe frame · compensated'
          ) : (
            <>
              {/* The tag carries the caveat at a glance. The sentence below
                  explains it, but a sentence is read once and a plate is read
                  every few seconds. */}
              <span className="tag tag--off">RAW</span> payload not removed
            </>
          )}
        </span>
      </div>
      <div className={`plate__body ${styles.contactBody}`}>
        <div className={styles.contactGrid}>
          {rows.map(([label, value, unit]) => (
            <div className={styles.contactCell} key={label}>
              <span className={styles.contactLabel}>{label}</span>
              <span className={`num ${styles.contactValue}`}>{value}</span>
              <span className={styles.contactUnit}>{unit}</span>
            </div>
          ))}
        </div>
        <div className={styles.contactZero}>
          <button type="button" onClick={() => onZero()} disabled={!wrench}>
            Zero
          </button>
          <button type="button" onClick={onClear} disabled={zero === null && !staleZero}>
            Clear
          </button>
          <span className={styles.contactZeroNote}>
            {zero
              ? `offset ${Math.hypot(...zero.vector).toFixed(2)} N · display only, the robot's own thresholds are untared`
              : staleZero
                ? 'zero was taken on the other pipeline stage — take it again'
                : 'takes the present reading as zero, with nothing touching the probe'}
          </span>
        </div>
        {!compensated ? (
          <p className={styles.contactNote}>
            No calibration loaded — these carry the tool's own weight, so read the
            total as an upper bound. Shear and the angle are the useful part until
            then. The electronic zero on the Calibration page is the one that also
            reaches the robot.
          </p>
        ) : null}
      </div>
    </section>
  );
}

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
                {/* One decimal on force.
                    Three decimals moved every frame on sensor noise alone, and
                    a digit that never holds still is not a reading — the eye
                    tracks the churn instead of the value. The limits it is
                    judged against are whole newtons. Torque keeps more places:
                    its whole signal lives in the thousandths. */}
                <td className={`num ${styles.wide} ${styles.fixed}`}>
                  {signed(value, unit === 'N' ? 1 : 4)}
                </td>
                <td className={styles.unit}>{unit}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {direct ? (
          <div className={styles.sensorLink}>
            {/* The sensor's own sampling rate, quantised so it stops wandering
                between 998 and 1003. This is a link property — whether the
                stream is healthy — and the panel it sits under updates once a
                second, each reading the mean of that second. */}
            <span className={styles.sensorItem}>
              <span className={styles.sensorLabel}>SENSOR</span>
              <span className={`num ${styles.rate}`}>{quantise(wrench?.sensorHz, 50)} Hz</span>
            </span>
            <span className={`${styles.sensorItem} ${badLink ? styles.sensorBad : ''}`}>
              <span className={styles.sensorLabel}>CRC</span>
              <span className={`num ${styles.rate}`}>{wrench?.crcErrors ?? 0}</span>
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
  acknowledged: { at: number } | null;
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
              ? formatClock(acknowledged.at)
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
