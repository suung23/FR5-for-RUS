import { useEffect, useState } from 'react';
import { config } from '../telemetry/config';
import type { ForceLimits, RobotTelemetry } from '../telemetry/types';
import styles from './ForceLimits.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

/** The bridge's rules, mirrored so a value it would refuse is caught here first. */
const MIN_FLOOR_N = 0.5;
const MIN_WINDOW_N = 1.0;
const WARN_GAP_N = 0.5;
const STEP_N = 0.1;

type Field = 'floor' | 'max';

/**
 * Contact-force window — the contact floor and the hard limit, nothing else.
 *
 * Probing mode is entered and left by the *Policy inference* buttons, not by a
 * force threshold (`teleop.contact_probing_force_trigger` is off). So there is
 * no entry force here. What the operator sets is what the probe does **once it
 * is in that mode**:
 *
 * - **Min contact** is the least force the probe keeps on the abdomen. The
 *   regulator's target is placed one deadband above it, so the moment the
 *   reading falls to the floor the robot pushes in. It is a line the controller
 *   reacts at, not a wall: the phantom is viscoelastic and force follows
 *   penetration by a second or two, so a breath can dip under it briefly.
 * - **Limit** is the force at which the stack retreats regardless.
 *
 * Warn (rising refused) sits a fixed half newton under the limit and the teleop
 * resume line follows the floor; both are derived by the bridge, because four
 * coupled numbers with invariants between them is a thing to compute, not to
 * type.
 *
 * **The readback row is the truth, the inputs are a request.** The plate shows
 * what the control stack reports holding now, read back from its parameters,
 * and the inputs hold the operator's draft until APPLY. A request the stack
 * refuses shows the refusal and leaves the readback unchanged.
 *
 * The limit cannot be raised above the ceiling the bridge was started with
 * (`bridge.force_limit_ceiling_n`, 10 N by default since 2026-10-19).
 */
export function ForceLimitsPlate({ telemetry, available, onCommand }: Props) {
  const live: ForceLimits | undefined = available ? telemetry.forceLimits : undefined;
  const ceiling = live?.ceilingN ?? config.maxForceN;
  const band = live?.deadbandN ?? config.targetBandN;

  const [floorText, setFloorText] = useState('');
  const [maxText, setMaxText] = useState('');
  const [seeded, setSeeded] = useState(false);
  const [sent, setSent] = useState<{ floorN: number; maxN: number } | null>(null);

  // Seed the draft from the first readback, and only once: after that the
  // inputs are the operator's and a frame must not overwrite what they are
  // typing. RESET puts the readback back into them deliberately.
  useEffect(() => {
    if (seeded || !live) return;
    setFloorText(live.floorN.toFixed(1));
    setMaxText(live.maxN.toFixed(1));
    setSeeded(true);
  }, [live, seeded]);

  const floor = Number(floorText);
  const max = Number(maxText);
  const problem = validate(floor, max, band, ceiling);
  const dirty = live
    ? Math.abs(floor - live.floorN) > 1e-6 || Math.abs(max - live.maxN) > 1e-6
    : true;
  const canApply = available && live !== undefined && problem === null && dirty;

  // "Requested, not confirmed" until the readback matches what was sent.
  const confirmed =
    sent !== null &&
    live !== undefined &&
    Math.abs(live.floorN - sent.floorN) < 1e-3 &&
    Math.abs(live.maxN - sent.maxN) < 1e-3;

  const apply = () => {
    if (!canApply) return;
    const request = { floorN: round(floor), maxN: round(max) };
    if (onCommand({ command: 'force.limits', ...request })) setSent(request);
  };

  const bump = (which: Field, dir: 1 | -1) => {
    const current = which === 'floor' ? floor : max;
    const next = round((Number.isFinite(current) ? current : 0) + dir * STEP_N);
    (which === 'floor' ? setFloorText : setMaxText)(next.toFixed(1));
  };

  const reset = () => {
    if (!live) return;
    setFloorText(live.floorN.toFixed(1));
    setMaxText(live.maxN.toFixed(1));
  };

  const inProbing = telemetry.probingMode !== undefined && telemetry.probingMode !== 'approach';

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Contact force window</span>
        <span className="plate__aside">
          {!available
            ? 'no source'
            : !live
              ? 'stack does not report its window'
              : sent === null
                ? 'as the stack holds it'
                : confirmed
                  ? 'applied'
                  : 'requested — not confirmed'}
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.entries}>
          <Entry
            label="Min contact"
            value={floorText}
            onChange={setFloorText}
            onStep={(d) => bump('floor', d)}
            disabled={!available || !live}
            invalid={problem?.field === 'floor'}
            min={MIN_FLOOR_N}
            max={ceiling - MIN_WINDOW_N - band}
          />
          <Entry
            label="Limit (max)"
            value={maxText}
            onChange={setMaxText}
            onStep={(d) => bump('max', d)}
            disabled={!available || !live}
            invalid={problem?.field === 'max'}
            min={MIN_FLOOR_N + MIN_WINDOW_N + band}
            max={ceiling}
          />
        </div>

        <div className={styles.readback}>
          <span className="field__label">Stack holds</span>
          <span className={`${styles.readbackValue} num`}>
            {live
              ? `min ${live.floorN.toFixed(2)} · target ${live.holdN.toFixed(2)} · warn ${live.warnN.toFixed(2)} · limit ${live.maxN.toFixed(2)} N`
              : '—'}
          </span>
        </div>

        <div className={styles.actions}>
          <div>
            <button type="button" onClick={reset} disabled={!live || !dirty}>
              Reset
            </button>
          </div>
          {problem ? <span className={styles.reason}>{problem.reason}</span> : null}
          <button type="button" className={styles.apply} onClick={apply} disabled={!canApply}>
            Apply
          </button>
        </div>

        <p className={styles.note}>
          Probing mode is entered with the policy START button, not by force. Once in it,
          the robot pushes in whenever contact falls to Min contact (regulator target{' '}
          {band.toFixed(2)} N above it) and retreats at Limit. Warn is {WARN_GAP_N.toFixed(1)} N
          under the limit; rising is refused above it. Ceiling {ceiling.toFixed(1)} N is fixed in
          the stack&apos;s configuration, not here.
          {live && inProbing ? ' The arm is probing: a new floor moves it now.' : ''}
        </p>
      </div>
    </section>
  );
}

function Entry({
  label,
  value,
  onChange,
  onStep,
  disabled,
  invalid,
  min,
  max,
}: {
  label: string;
  value: string;
  onChange(next: string): void;
  onStep(dir: 1 | -1): void;
  disabled: boolean;
  invalid: boolean;
  min: number;
  max: number;
}) {
  return (
    <label className={styles.entry}>
      <span className={styles.entryLabel}>{label}</span>
      <span className={styles.entryRow}>
        <input
          className={`${styles.input} num`}
          type="number"
          inputMode="decimal"
          step={STEP_N}
          min={min}
          max={max}
          value={value}
          disabled={disabled}
          aria-invalid={invalid}
          onChange={(e) => onChange(e.target.value)}
        />
        <span className={styles.unit}>N</span>
      </span>
      <span className={styles.steps}>
        <button type="button" className={styles.step} disabled={disabled} onClick={() => onStep(-1)}>
          &minus;
        </button>
        <button type="button" className={styles.step} disabled={disabled} onClick={() => onStep(1)}>
          +
        </button>
      </span>
    </label>
  );
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}

function validate(
  floor: number,
  max: number,
  band: number,
  ceiling: number,
): { field: Field; reason: string } | null {
  if (!Number.isFinite(floor)) return { field: 'floor', reason: 'Min contact must be a number.' };
  if (!Number.isFinite(max)) return { field: 'max', reason: 'Limit must be a number.' };
  if (floor < MIN_FLOOR_N) {
    return { field: 'floor', reason: `Min contact cannot go below ${MIN_FLOOR_N.toFixed(1)} N.` };
  }
  if (max > ceiling + 1e-9) {
    return { field: 'max', reason: `Limit cannot exceed the ${ceiling.toFixed(1)} N ceiling.` };
  }
  if (floor + band > max - MIN_WINDOW_N + 1e-9) {
    return {
      field: 'floor',
      reason: `Min contact must sit at least ${(MIN_WINDOW_N + band).toFixed(2)} N under the limit.`,
    };
  }
  return null;
}
