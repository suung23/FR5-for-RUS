import { useState } from 'react';
import type { RobotTelemetry } from '../telemetry/types';
import styles from './PolicyInference.module.css';

interface Props {
  telemetry: RobotTelemetry;
  available: boolean;
  onCommand(command: Record<string, unknown>): boolean;
}

/**
 * Policy inference — start and stop, and nothing else.
 *
 * The two buttons send `policy.enable`, which is a **permission for the policy
 * runner**, not a motion command and not a mode change. It is deliberately
 * separate from contact probing: that transition is a safety judgement made
 * from contact force, and it carries the velocity clamp (150 mm/s free space
 * down to 10 mm/s on contact). Tying the two to one button would mean stopping
 * the policy by releasing that clamp, which is the wrong thing to do at the
 * moment you want the arm to slow down.
 *
 * So the stack keeps deciding when contact happened. This panel only decides
 * whether the policy may drive within whatever limits the stack has set.
 *
 * **The fill follows the mode that came back, not the button.** When the request
 * reaches the control stack it takes the contact regime and declares
 * `contact_probing_policy` on `probing_mode` — the same rule the operator
 * station and the in-plane toggle follow. A request that never arrives leaves
 * the panel unlit, which is the honest reading: the stack may not be up.
 *
 * The sub-mode keeps the `contact_probing` prefix on purpose. Every consumer
 * tests that prefix — `us_servo_node` stops discarding the small commands the
 * policy produces, and the runner reads the same string as its handover.
 *
 * Commanding stays inside the force envelope the stack owns: the hold target is
 * 3 N, rising is refused at 4.5 N, and 5 N (`safety.max_contact_force_n`) forces
 * a retreat. The policy stops on that retreat.
 */
export function PolicyInference({ telemetry, available, onCommand }: Props) {
  const [requested, setRequested] = useState<boolean | null>(null);
  const mode = telemetry.probingMode;
  const known = mode !== undefined;
  // The stack declares this when the policy holds the regime. That is the confirmation.
  const held = mode === 'contact_probing_policy';
  const contact = known && mode !== 'approach';

  const send = (enabled: boolean) => {
    if (onCommand({ command: 'policy.enable', enabled })) setRequested(enabled);
  };

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Policy inference</span>
        <span className="plate__aside">
          {held
            ? 'policy holds the contact regime'
            : requested === null
              ? 'no request sent'
              : requested
                ? 'start requested \u2014 not confirmed'
                : 'stopped'}
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.choices} role="group" aria-label="Policy inference">
          <button
            type="button"
            aria-pressed={held}
            disabled={!available}
            className={`${styles.choice} ${held ? styles.applied : ''} ${
              requested === true && !held ? styles.pending : ''
            }`}
            onClick={() => send(true)}
          >
            <span className={styles.choiceLabel}>Start inference</span>
            <span className={styles.choiceNote}>policy may drive within the stack&rsquo;s limits</span>
          </button>
          <button
            type="button"
            aria-pressed={known && !held}
            disabled={!available}
            className={`${styles.choice} ${known && !held ? styles.applied : ''}`}
            onClick={() => send(false)}
          >
            <span className={styles.choiceLabel}>Stop</span>
            <span className={styles.choiceNote}>runner goes idle · contact judgement untouched</span>
          </button>
        </div>

        <p className={styles.state} role="status">
          {held ? (
            <>
              <span className="tag tag--strong">HANDOVER</span> the stack holds the beam axis for
              the policy and the operator&rsquo;s axes are zeroed.{' '}
              <strong>Let go of the stylus</strong> — one publisher per command channel.
            </>
          ) : requested === true ? (
            <>
              <span className="tag tag--strong">PENDING</span> the request went out but the stack
              has not declared <code>contact_probing_policy</code>. Either it is not up, or it
              refused. Nothing has changed on the robot.
            </>
          ) : !known ? (
            <>
              The control stack has not declared a mode. Starting is still allowed — the runner
              holds its own contact condition and the stack owns the force envelope.
            </>
          ) : contact ? (
            <>
              Contact regime, held by the force judgement rather than the policy. Starting here
              hands the same regime to the policy.
            </>
          ) : (
            <>
              Not probing. Whatever the mode, the force envelope is the stack&rsquo;s:
              5&nbsp;N forces a retreat and the policy stops with it.
            </>
          )}
        </p>
      </div>
    </section>
  );
}
