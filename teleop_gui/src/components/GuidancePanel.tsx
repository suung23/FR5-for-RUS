import type { GuidanceItem } from '../telemetry/guidance';
import styles from './GuidancePanel.module.css';

interface Props {
  items: GuidanceItem[];
  /** How many to show. The rest stay out of the way; a guidance list long
   *  enough to scroll is a list nobody reads. */
  limit?: number;
}

export function GuidancePanel({ items, limit = 4 }: Props) {
  const shown = items.slice(0, limit);
  const hidden = items.length - shown.length;

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Operator guidance</span>
        {hidden > 0 ? <span className={styles.more}>+{hidden} more</span> : null}
      </div>
      <div className={`panel__body ${styles.body}`}>
        <ul className={styles.list}>
          {shown.map((item) => (
            <li key={item.id} className={`${styles.item} ${styles[item.tone]}`}>
              <span className={styles.bullet} aria-hidden="true" />
              <span className={styles.text}>{item.text}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
