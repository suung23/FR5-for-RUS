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
 * **Nothing here is a confirmation.** Unlike the probing mode, the runner
 * publishes no state the console can read back, so the panel prints what it
 * asked for and says so. A lit button that meant "inferring" would be a claim
 * the console cannot support — the runner may not be up at all.
 *
 * Commanding stays inside the force envelope the stack owns: the hold target is
 * 3 N, rising is refused at 4.5 N, and 5 N (`safety.max_contact_force_n`) forces
 * a retreat. The policy stops on that retreat.
 */
export function PolicyInference({ telemetry, available, onCommand }: Props) {
  const [requested, setRequested] = useState<boolean | null>(null);
  const mode = telemetry.probingMode;
  const known = mode !== undefined;
  const probing = mode === 'contact_probing' || mode === 'contact_probing_inplane';

  const send = (enabled: boolean) => {
    if (onCommand({ command: 'policy.enable', enabled })) setRequested(enabled);
  };

  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Policy inference</span>
        <span className="plate__aside">
          {requested === null ? 'no request sent' : requested ? 'start requested' : 'stop requested'}
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        <div className={styles.choices} role="group" aria-label="Policy inference">
          <button
            type="button"
            aria-pressed={requested === true}
            disabled={!available}
            className={`${styles.choice} ${requested === true ? styles.applied : ''}`}
            onClick={() => send(true)}
          >
            <span className={styles.choiceLabel}>Start inference</span>
            <span className={styles.choiceNote}>policy may drive within the stack&rsquo;s limits</span>
          </button>
          <button
            type="button"
            aria-pressed={requested === false}
            disabled={!available}
            className={`${styles.choice} ${requested === false ? styles.applied : ''}`}
            onClick={() => send(false)}
          >
            <span className={styles.choiceLabel}>Stop</span>
            <span className={styles.choiceNote}>runner goes idle · contact judgement untouched</span>
          </button>
        </div>

        <p className={styles.state} role="status">
          {requested === true ? (
            <>
              <span className="tag tag--strong">REQUESTED</span> the console cannot confirm the
              runner is up — it publishes no state to read back. If it is running and there is
              contact, <strong>let go of the stylus</strong>: one publisher per command channel.
            </>
          ) : !known ? (
            <>
              The control stack has not declared a mode. Starting is still allowed — the runner
              holds its own contact condition and the stack owns the force envelope.
            </>
          ) : probing ? (
            <>
              Contact probing — the robot holds the beam axis and the contact velocity limits
              apply. This is the window the policy is meant to run in.
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
