import { useEffect, useRef } from 'react';
import { BRIEFING_SECTIONS } from '../telemetry/guidanceGate';
import styles from './GuidanceGate.module.css';

interface Props {
  onAcknowledge(): void;
}

/**
 * Full-screen guidance gate.
 *
 * Shown once per session. Blocks the console until the operator
 * acknowledges, and does not return. The console stays visible
 * behind it and keeps updating — the point is that the operator reads the
 * checklist while looking at the live state it refers to, not at a blank
 * screen.
 *
 * Enter is the only dismissal. There is no timeout and no click-away: an
 * overlay that closes itself has not been read, and one that closes on a stray
 * click can be dismissed without anyone noticing.
 *
 * Focus is trapped inside the panel. Tab cannot reach the console behind it, so
 * a keystroke meant for the acknowledgement cannot land somewhere else.
 */
export function GuidanceGate({ onAcknowledge }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    buttonRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        onAcknowledge();
        return;
      }
      // Trap focus. The panel holds exactly one focusable control, so any Tab
      // simply returns to it rather than walking into the inert console.
      if (event.key === 'Tab') {
        event.preventDefault();
        buttonRef.current?.focus();
      }
    };
    // Capture phase: the gate sees the key before anything behind it can.
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [onAcknowledge]);

  return (
    <div
      className={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-labelledby="guidance-title"
    >
      <div className={styles.panel} ref={panelRef}>
        <p className={styles.kicker}>Operator guidance</p>
        <h2 className={styles.title} id="guidance-title">
          Before you begin
        </h2>
        <div className={styles.rule} />

        {BRIEFING_SECTIONS.map((section) => (
          <section className={styles.section} key={section.title}>
            <h3 className={styles.sectionTitle}>{section.title}</h3>
            <ol className={styles.list}>
              {section.items.map((item, i) => (
                <li key={i}>
                  <span className={styles.index}>{i + 1}.</span>
                  <span className={styles.itemText}>{item}</span>
                </li>
              ))}
            </ol>
          </section>
        ))}

        <p className={styles.footer}>
          Review the instructions, then press ENTER to continue.
        </p>

        <button type="button" className={styles.action} ref={buttonRef} onClick={onAcknowledge}>
          <span className={styles.key}>[ ENTER ]</span>
          <span className={styles.actionText}>Continue</span>
        </button>
      </div>
    </div>
  );
}
