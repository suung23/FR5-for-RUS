/**
 * Approach / contact classification from the contact force.
 *
 * A TypeScript port of `fr5_ik/probing_mode.py`, kept deliberately faithful so
 * the panel on screen and the node in the control stack can never disagree
 * about what "contact" means. If one changes, change both.
 *
 * **What crosses the threshold is the caller's decision, not this class's**
 * (2026-08-31). It takes one scalar, positive when pressing, and compares it.
 * The control stack picks between the contact-force magnitude and the normal
 * component in `ft_sensor.contact_force_mode`, and the console has to feed in
 * whichever that is — a classifier holding its own opinion about which scalar
 * counts is a classifier that will eventually disagree with the arm.
 *
 * It used to take a probe-frame `F_z` and apply `normalForceSign` itself. That
 * baked the normal component in, which stopped being right when the stack
 * moved to `‖F‖` at a 1 N threshold: a probe touching even slightly off-axis
 * puts most of the contact into shear, and `F_z` alone reads that as no
 * contact at all.
 *
 * Three behaviours matter, and none of them are optional:
 *
 * - **Hysteresis.** Separate enter and release thresholds. A single threshold
 *   chatters at the boundary, and that chatter is a mode flag other software
 *   reads.
 * - **Confirmation windows.** A threshold crossing has to persist. Release is
 *   confirmed far more slowly than entry, because a momentary dip in force
 *   while still pressing must not read as "clear".
 * - **Latch.** `hasContacted` remembers that contact happened at all. Anything
 *   that should not be undone by briefly lifting the probe reads the latch,
 *   not the instantaneous state.
 */

export type ContactPhase = 'approach' | 'contact';

export interface ContactOptions {
  enterN: number;
  releaseN: number;
  /** Constant offset subtracted from the reading, in newtons. */
  biasN?: number;
  confirmMs?: number;
  releaseConfirmMs?: number;
}

export interface ContactSnapshot {
  phase: ContactPhase;
  hasContacted: boolean;
  /** The scalar the judgement was made on. Positive = pressing. */
  contactForceN: number;
  /** 0..1 progress through the active confirmation window. */
  confirmProgress: number;
}

export class ContactDetector {
  private readonly opts: Required<ContactOptions>;
  private phase: ContactPhase = 'approach';
  private latched = false;
  private contactForceN = 0;
  private aboveMs = 0;
  private belowMs = 0;

  constructor(options: ContactOptions) {
    if (options.releaseN >= options.enterN) {
      throw new Error(
        `releaseN (${options.releaseN}) must be below enterN (${options.enterN}); ` +
          'without hysteresis the phase chatters at the threshold',
      );
    }
    // probe.yaml: teleop.contact_probing_confirm_s = 0.02,
    // teleop.contact_probing_release_confirm_s = 0.5. Release is confirmed far
    // more slowly than entry, because a momentary dip while still pressing
    // must not read as "clear".
    this.opts = {
      biasN: 0,
      confirmMs: 20,
      releaseConfirmMs: 500,
      ...options,
    };
  }

  reset(unlatch = false): void {
    this.phase = 'approach';
    this.aboveMs = 0;
    this.belowMs = 0;
    if (unlatch) this.latched = false;
  }

  /**
   * Feed one sample.
   *
   * @param contactForceN Contact force in newtons, positive when pressing.
   *   Which scalar that is — `‖F‖` or `F_n` — is the caller's to decide.
   */
  update(contactForceN: number, dtMs: number): ContactSnapshot {
    const fn = contactForceN - this.opts.biasN;
    this.contactForceN = fn;
    const dt = Math.max(0, dtMs);

    if (this.phase === 'approach') {
      if (fn >= this.opts.enterN) {
        this.aboveMs += dt;
        if (this.aboveMs >= this.opts.confirmMs) {
          this.phase = 'contact';
          this.latched = true;
          this.belowMs = 0;
        }
      } else {
        // A single sample back under the threshold restarts the window.
        // "Mostly above" is not contact.
        this.aboveMs = 0;
      }
    } else if (fn <= this.opts.releaseN) {
      this.belowMs += dt;
      if (this.belowMs >= this.opts.releaseConfirmMs) {
        this.phase = 'approach';
        this.aboveMs = 0;
      }
    } else {
      this.belowMs = 0;
    }

    return this.snapshot();
  }

  snapshot(): ContactSnapshot {
    const window =
      this.phase === 'approach' ? this.opts.confirmMs : this.opts.releaseConfirmMs;
    const elapsed = this.phase === 'approach' ? this.aboveMs : this.belowMs;
    return {
      phase: this.phase,
      hasContacted: this.latched,
      contactForceN: this.contactForceN,
      confirmProgress: window > 0 ? Math.min(1, elapsed / window) : elapsed > 0 ? 1 : 0,
    };
  }

  /**
   * Whether freespace velocity scaling is still appropriate.
   *
   * Reads the latch rather than the phase. Once the probe has touched
   * something, lifting it does not put the session back into an approach.
   */
  get freespaceAllowed(): boolean {
    return !this.latched;
  }
}
