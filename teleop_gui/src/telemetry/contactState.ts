/**
 * Approach / contact classification from the normal force.
 *
 * A TypeScript port of `fr5_control/contact_state.py`, kept deliberately
 * faithful so the panel on screen and the node in the control stack can never
 * disagree about what "contact" means. If one changes, change both.
 *
 * Sign convention (DESIGN_NOTES §4.4):
 *
 *     F_n = normalForceSign x F_z^probe        positive = compression
 *
 * With `normalForceSign = -1`, compression arrives as a negative `F_z`, so the
 * `F_z <= -7 N` boundary the operator cares about is `F_n >= 7 N` here.
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
  normalForceSign: number;
  /** Constant offset subtracted from F_n, in newtons. */
  biasN?: number;
  confirmMs?: number;
  releaseConfirmMs?: number;
}

export interface ContactSnapshot {
  phase: ContactPhase;
  hasContacted: boolean;
  normalForceN: number;
  /** 0..1 progress through the active confirmation window. */
  confirmProgress: number;
}

export class ContactDetector {
  private readonly opts: Required<ContactOptions>;
  private phase: ContactPhase = 'approach';
  private latched = false;
  private normalForceN = 0;
  private aboveMs = 0;
  private belowMs = 0;

  constructor(options: ContactOptions) {
    if (options.releaseN >= options.enterN) {
      throw new Error(
        `releaseN (${options.releaseN}) must be below enterN (${options.enterN}); ` +
          'without hysteresis the phase chatters at the threshold',
      );
    }
    this.opts = {
      biasN: 0,
      confirmMs: 5,
      releaseConfirmMs: 50,
      ...options,
    };
  }

  /** Convert a probe-frame F_z into the signed normal force. */
  normalForce(fz: number): number {
    return this.opts.normalForceSign * fz - this.opts.biasN;
  }

  reset(unlatch = false): void {
    this.phase = 'approach';
    this.aboveMs = 0;
    this.belowMs = 0;
    if (unlatch) this.latched = false;
  }

  update(fz: number, dtMs: number): ContactSnapshot {
    const fn = this.normalForce(fz);
    this.normalForceN = fn;
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
      normalForceN: this.normalForceN,
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
