import { useState } from 'react';
import type { CalibrationStatus, CompensatedStages, RobotTelemetry, WrenchSample } from '../telemetry/types';
import styles from './SensorCalibration.module.css';

interface Props {
  telemetry: RobotTelemetry;
  wrench: WrenchSample | null;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

const AXES = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'];

/**
 * Sensor calibration and frame registration.
 *
 * Two procedures, deliberately separate. The electronic zero captures channel
 * bias with the assembly hanging free at one pose; the multi-pose fit solves
 * the payload mass and centre of mass so the gravity wrench can be predicted at
 * any pose. A zero alone is only correct at the pose it was taken.
 *
 * Every stage of the pipeline is shown side by side. Which stage a reading goes
 * wrong at is which calibration is wrong, and that cannot be seen from the
 * final number alone.
 *
 * **The robot is never moved from here.** The operator moves it to each pose by
 * hand and then presses capture; the bridge refuses anything else.
 */
export function SensorCalibration({ telemetry, wrench, available, onCommand }: Props) {
  const calib = telemetry.calibration;
  const [note, setNote] = useState('');
  const [angle, setAngle] = useState<string>('');

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Sensor calibration</span>
        <span className="plate__aside">
          {calib?.present
            ? calib.valid
              ? `Valid · ${(calib.ageHours ?? 0).toFixed(1)} h old`
              : 'Invalid — contact control blocked'
            : 'No profile'}
        </span>
      </div>

      <div className={`plate__body ${styles.body}`}>
        <div className={styles.col}>
          <Checklist />
          <Procedures calib={calib} onCommand={onCommand} note={note} setNote={setNote}
                      angle={angle} setAngle={setAngle} />
        </div>

        <div className={styles.col}>
          <Pipeline wrench={wrench} available={available} />
          <Registration calib={calib} telemetry={telemetry} />
          <Quality calib={calib} />
        </div>
      </div>
    </section>
  );
}

/** Installation checks the operator confirms before anything is captured. */
function Checklist() {
  const items = [
    'PX6D is rigidly mounted between the FR5 flange and the probe holder.',
    'All mount screws are tight; the probe cannot shift relative to the sensor.',
    'The cable is strain-relieved and applies no variable force or torque.',
    'Probe, mount, cable, and sensor touch nothing — phantom, table, robot, or operator.',
    'The probe’s intended normal/contact direction is documented.',
  ];
  return (
    <div className={styles.block}>
      <h3 className={styles.subhead}>Installation checks</h3>
      <ol className={styles.checklist}>
        {items.map((item, i) => (
          <li key={i}>
            <span className={styles.index}>{i + 1}.</span>
            <span>{item}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Procedures({
  calib, onCommand, note, setNote, angle, setAngle,
}: {
  calib: CalibrationStatus | undefined;
  onCommand(c: Record<string, unknown>): boolean;
  note: string;
  setNote(v: string): void;
  angle: string;
  setAngle(v: string): void;
}) {
  const session = calib?.session;
  const capturing = session?.capturing ?? '';
  const poses = session?.poseCount ?? 0;
  const target = session?.poseTarget ?? 12;

  return (
    <div className={styles.block}>
      <h3 className={styles.subhead}>A · Electronic zero</h3>
      <p className={styles.prose}>
        Capture with the complete probe assembly attached and touching nothing. This
        alone is <strong>not sufficient</strong>: the mounted probe and holder produce
        an orientation-dependent gravity wrench, so a zero taken at one pose drifts by
        several newtons at another.
      </p>
      <div className={styles.row}>
        <button type="button" disabled={!!capturing}
                onClick={() => onCommand({ command: 'calib.bias', seconds: 4.0 })}>
          Capture zero (4 s)
        </button>
        <span className={styles.status}>
          {session?.biasAccepted === null || session?.biasAccepted === undefined
            ? 'not captured'
            : session.biasAccepted
              ? `accepted — ${session.biasReason}`
              : `rejected — ${session.biasReason}`}
        </span>
      </div>

      <h3 className={styles.subhead}>B · Multi-pose gravity</h3>
      <p className={styles.prose}>
        Move the robot by hand to each pose, wait for it to settle, then capture. Use at
        least {target} poses that point gravity in substantially different directions in
        the sensor frame — poses that all look alike leave the centre of mass
        unobservable, and the residual will not reveal it.
        {' '}With auto on, drag the arm and let go: each rest records itself. A rest too
        close to one already taken is skipped, and the line below says by how much it
        missed. Once there are enough poses to solve, the model is fitted and saved
        without a further press — and refitted and resaved as more poses come in, since
        each one can only improve it. Only a valid fit is ever written, and the previous
        calibration is kept beside it.
      </p>
      <div className={styles.row}>
        {/* Auto-capture exists because posing the arm takes both hands. Pressing
            a button between every pose means letting go of the robot, and the
            arm is what the operator is holding. */}
        <button type="button"
                onClick={() => onCommand({
                  command: 'calib.auto',
                  enabled: !calib?.autoCapture,
                  seconds: 2.0,
                })}>
          {calib?.autoCapture ? 'Auto: capture + save ON' : 'Auto: capture + save off'}
        </button>
        <button type="button" disabled={!!capturing}
                onClick={() => onCommand({ command: 'calib.pose', seconds: 2.0 })}>
          Capture pose
        </button>
        <button type="button" disabled={!!capturing || poses === 0}
                onClick={() => onCommand({ command: 'calib.fit' })}>
          Fit model
        </button>
        <button type="button" disabled={!!capturing}
                onClick={() => onCommand({ command: 'calib.load' })}>
          Load recorded
        </button>
        <button type="button" disabled={!!capturing}
                onClick={() => onCommand({ command: 'calib.reset' })}>
          Discard
        </button>
      </div>
      <div className={styles.row}>
        <span className={styles.status}>
          {poses} / {target} poses · coverage {(session?.coverage ?? 0).toFixed(3)}
          {calib?.autoCapture
            ? calib?.autoArmed
              ? ' · auto: armed'
              : ' · auto: waiting for movement'
            : ''}
        </span>
        {capturing ? (
          <span className={styles.status}>
            capturing {session?.captureLabel} — {((session?.captureProgress ?? 0) * 100).toFixed(0)} %
          </span>
        ) : null}
      </div>

      <h3 className={styles.subhead}>Save profile</h3>
      <div className={styles.row}>
        <label className={styles.field}>
          Mounting angle
          <input value={angle} onChange={(e) => setAngle(e.target.value)}
                 placeholder={String(calib?.mountingAngleDeg ?? 45)} inputMode="decimal" />
          <span className={styles.unit}>deg</span>
        </label>
      </div>
      <div className={styles.row}>
        <label className={styles.field}>
          Note
          <input value={note} onChange={(e) => setNote(e.target.value)}
                 placeholder="probe and mount configuration" />
        </label>
      </div>
      <div className={styles.row}>
        <button type="button" onClick={() => onCommand({
          command: 'calib.save',
          note,
          ...(angle.trim() ? { mountingAngleDeg: Number(angle) } : {}),
        })}>
          Save calibration
        </button>
        {/* The working-pose zero sits on top of the multi-pose compensation.
            It is a constant, so it is exact only at the pose it was taken —
            which is why the bridge insists on the working pose and reports how
            far the arm has since moved from it. */}
        <button type="button" onClick={() => onCommand({ command: 'calib.tare' })}>
          Zero at working pose
        </button>
        {calib?.workingTare ? (
          <button type="button" onClick={() => onCommand({ command: 'calib.tare.clear' })}>
            Clear zero
          </button>
        ) : null}
        <span className={styles.status}>{session?.lastError || ''}</span>
      </div>
    </div>
  );
}

/** Raw → bias-corrected → gravity-compensated → probe frame → contact point. */
function Pipeline({ wrench, available }: { wrench: WrenchSample | null; available: boolean }) {
  const stages = available ? wrench?.compensated : undefined;
  const rows: [string, keyof CompensatedStages][] = [
    ['Raw {S}', 'rawSensor'],
    ['Bias corrected {S}', 'biasCorrectedSensor'],
    ['External {S}', 'externalSensor'],
    ['External {P}', 'externalProbe'],
    ['Contact {P}', 'contactProbe'],
  ];

  return (
    <div className={styles.block}>
      <h3 className={styles.subhead}>Compensation pipeline</h3>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Stage</th>
            {AXES.map((a) => <th key={a} scope="col">{a}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, key]) => {
            const values = stages ? (stages[key] as number[]) : undefined;
            return (
              <tr key={key} className={key === 'contactProbe' ? styles.finalRow : undefined}>
                <th scope="row">{label}</th>
                {AXES.map((axis, i) => (
                  <td key={axis} className="num">
                    {values ? values[i].toFixed(i < 3 ? 3 : 4) : '—'}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className={styles.note}>
        {stages
          ? `Normal force F(zP) = ${stages.normalForceN.toFixed(3)} N — positive is compression.`
          : 'No calibration profile loaded; only raw sensor values are available.'}
      </p>
    </div>
  );
}

function Registration({
  calib, telemetry,
}: {
  calib: CalibrationStatus | undefined;
  telemetry: RobotTelemetry;
}) {
  const rot = calib?.rotationProbeFromSensor;
  const q = telemetry.tcpPose?.quaternion;
  return (
    <div className={styles.block}>
      <h3 className={styles.subhead}>Frame registration</h3>
      <div className="field">
        <span className="field__label">Mounting angle</span>
        <span className="field__value num">
          {calib ? `${calib.mountingAngleDeg.toFixed(1)} deg` : '—'}
          {calib?.axialFlip ? ' · axial flip' : ''}
        </span>
      </div>
      <div className="field">
        <span className="field__label">Lever r(S→P)</span>
        <span className="field__value num">
          {calib ? calib.leverSensorToProbeM.map((v) => (v * 1000).toFixed(1)).join(' ') + ' mm' : '—'}
        </span>
      </div>
      <div className="field">
        <span className="field__label">FR5 orientation</span>
        <span className="field__value num">
          {q ? q.map((v) => v.toFixed(3)).join(' ') : '—'}
        </span>
      </div>
      <table className={styles.matrix}>
        <tbody>
          {(rot ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]]).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className="num">{rot ? v.toFixed(4) : '—'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className={styles.note}>
        Probe-frame rotation R(P←S). Raw sensor channels are never used as probe-frame
        channels; the sign convention lives in this matrix and nowhere else.
      </p>
    </div>
  );
}

function Quality({ calib }: { calib: CalibrationStatus | undefined }) {
  // Prefer the model the operator has just fitted, and fall back to the saved
  // profile.
  //
  // These two are different objects. `Fit model` puts its result on the
  // session; only `Save calibration` writes a profile. Reading the profile
  // alone meant that pressing Fit changed nothing on screen — the numbers
  // stayed at "—" until a save that the operator could not judge whether to
  // make. Fitting has to be visible before it can be accepted.
  const fitted = calib?.session?.gravity;
  const mass = fitted?.mass_kg ?? calib?.massKg;
  const com = fitted?.com_sensor_m ?? calib?.comSensorM;
  const rmsF = fitted?.rms_force_n ?? calib?.rmsForceN;
  const rmsT = fitted?.rms_torque_nm ?? calib?.rmsTorqueNm;
  const axisF = fitted?.per_axis_force_n ?? calib?.perAxisForceN;
  const axisT = fitted?.per_axis_torque_nm ?? calib?.perAxisTorqueNm;
  const coverage = fitted?.coverage ?? calib?.coverage;
  const poses = fitted?.poses ?? calib?.poses;

  const rows: [string, string][] = [
    ['Source', fitted ? 'fitted — not saved yet' : calib?.present ? 'saved profile' : '—'],
    ['Payload mass', mass != null ? `${(mass * 1000).toFixed(1)} g` : '—'],
    ['COM in {S}', com ? com.map((v) => (v * 1000).toFixed(1)).join(' ') + ' mm' : '—'],
    ['RMS force', rmsF != null ? `${rmsF.toFixed(3)} N` : '—'],
    ['RMS torque', rmsT != null ? `${rmsT.toFixed(4)} N·m` : '—'],
    ['Per-axis force', axisF ? axisF.map((v) => v.toFixed(3)).join(' ') : '—'],
    ['Per-axis torque', axisT ? axisT.map((v) => v.toFixed(4)).join(' ') : '—'],
    ['Pose coverage', coverage != null ? coverage.toFixed(3) : '—'],
    ['Poses used', poses != null ? String(poses) : '—'],
    ['Captured', calib?.createdAt ? new Date(calib.createdAt).toLocaleString('en-GB', { hour12: false }) : '—'],
    [
      'Working zero',
      calib?.workingTare
        ? `${Math.hypot(...calib.workingTare.slice(0, 3)).toFixed(3)} N removed`
        : 'not taken',
    ],
    [
      'Off tare pose',
      calib?.workingTare
        ? calib.tareOffAxisDeg != null
          ? `${calib.tareOffAxisDeg.toFixed(0)} deg`
          : '—'
        : '—',
    ],
  ];
  return (
    <div className={styles.block}>
      <h3 className={styles.subhead}>Calibration quality</h3>
      {rows.map(([label, value]) => (
        <div className="field" key={label}>
          <span className="field__label">{label}</span>
          <span className="field__value num">{value}</span>
        </div>
      ))}
      {calib?.session?.gravity && !calib.session.gravity.valid ? (
        <ul className={styles.issues}>
          {calib.session.gravity.issues.map((issue, i) => <li key={`fit-${i}`}>{issue}</li>)}
        </ul>
      ) : null}
      {calib?.issues?.length ? (
        <ul className={styles.issues}>
          {calib.issues.map((issue, i) => <li key={i}>{issue}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
