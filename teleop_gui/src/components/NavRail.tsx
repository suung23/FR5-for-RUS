import styles from './NavRail.module.css';

export type ConsoleView =
  | 'monitoring'
  | 'teleoperation'
  | 'segmentation'
  | 'calibration'
  | 'safety';

export const CONSOLE_VIEWS: ConsoleView[] = [
  'monitoring',
  'teleoperation',
  'segmentation',
  'calibration',
  'safety',
];

/** Read the view from the URL hash, so a reload lands where the operator was. */
export function viewFromHash(hash: string): ConsoleView | null {
  const name = hash.replace(/^#\/?/, '').toLowerCase();
  return (CONSOLE_VIEWS as string[]).includes(name) ? (name as ConsoleView) : null;
}

interface Props {
  view: ConsoleView;
  onSelect(view: ConsoleView): void;
  /** Count of unacknowledged warn/alarm events, shown beside SAFETY. */
  alarmCount: number;
}

/**
 * Left navigation rail.
 *
 * Operational views, not decorative icons. Selecting one changes the lower
 * workspace panel and the emphasis of the right column; the 3D view is common
 * to all of them because the arm's pose is never irrelevant.
 *
 * The selected item is the one place a filled navy block appears. Selection is
 * also carried by the left edge weight and by `aria-current`, so it does not
 * depend on the colour being visible.
 */
const ITEMS: { id: ConsoleView; label: string; note: string }[] = [
  { id: 'monitoring', label: 'Monitoring', note: 'Force trend' },
  { id: 'teleoperation', label: 'Teleoperation', note: 'Joint rates' },
  { id: 'segmentation', label: 'Segmentation', note: 'Bladder mask' },
  { id: 'calibration', label: 'Calibration', note: 'Sensor frames' },
  { id: 'safety', label: 'Safety', note: 'Limits · events' },
];

export function NavRail({ view, onSelect, alarmCount }: Props) {
  return (
    <nav className={styles.rail} aria-label="Console views">
      <div className={styles.railHead}>VIEW</div>
      <ul className={styles.list}>
        {ITEMS.map((item) => {
          const active = item.id === view;
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`${styles.item} ${active ? styles.itemActive : ''}`}
                onClick={() => onSelect(item.id)}
                aria-current={active ? 'page' : undefined}
              >
                <span className={styles.itemLabel}>{item.label}</span>
                <span className={styles.itemNote}>{item.note}</span>
                {item.id === 'safety' && alarmCount > 0 ? (
                  <span className={styles.count}>{alarmCount > 99 ? '99+' : alarmCount}</span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>

      <div className={styles.foot}>
        <div className={styles.footLine}>RESEARCH</div>
        <div className={styles.footLine}>USE ONLY</div>
      </div>
    </nav>
  );
}
