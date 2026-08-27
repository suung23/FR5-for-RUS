import { useCallback, useEffect, useRef, useState } from 'react';
import { EventLog } from './components/EventLog';
import { GuidanceGate } from './components/GuidanceGate';
import { ForceTrend } from './components/ForceTrend';
import { JointRates } from './components/JointRates';
import { ModeTimeline } from './components/ModeTimeline';
import { NavRail, viewFromHash, type ConsoleView } from './components/NavRail';
import { SafetyPanel } from './components/SafetyPanel';
import { StatusColumn } from './components/StatusColumn';
import { StatusStrip } from './components/StatusStrip';
import { Workspace } from './components/Workspace';
import { GateTrigger, type GateTopic } from './telemetry/guidanceGate';
import { isStale } from './store/selectors';
import { logEvent } from './store/events';
import { subscribeEvents } from './store/events';
import { useTelemetryStore } from './store/telemetryStore';
import styles from './App.module.css';

export function App() {
  const telemetry = useTelemetryStore((s) => s.telemetry);
  const wrench = useTelemetryStore((s) => s.wrench);
  const contact = useTelemetryStore((s) => s.contact);
  const link = useTelemetryStore((s) => s.link);
  const history = useTelemetryStore((s) => s.history);
  const trajectory = useTelemetryStore((s) => s.trajectory);
  const peak = useTelemetryStore((s) => s.peakNormalForceN);
  const frameRateHz = useTelemetryStore((s) => s.frameRateHz);
  const paused = useTelemetryStore((s) => s.paused);
  const setPaused = useTelemetryStore((s) => s.setPaused);
  const resetSession = useTelemetryStore((s) => s.resetSession);

  // The selected view lives in the URL hash. A reload — or a crash and restart
  // mid-procedure — returns to the panel the operator was on rather than to a
  // default they then have to re-select.
  // Guidance gate. One is owed at startup and on each transition that changes
  // what the operator must watch; the console stays inert until acknowledged.
  const triggerRef = useRef(new GateTrigger());
  const [gate, setGate] = useState<GateTopic | null>(() =>
    triggerRef.current.initialTopic(),
  );
  const [acknowledged, setAcknowledged] = useState<{ topic: GateTopic; at: number } | null>(
    null,
  );

  useEffect(() => {
    const owed = triggerRef.current.observe({
      robotState: telemetry.robotState,
      safetyState: telemetry.safetyState,
      probingMode: telemetry.probingMode,
    });
    if (owed) setGate(owed);
  }, [telemetry.robotState, telemetry.safetyState, telemetry.probingMode]);

  const acknowledge = useCallback(() => {
    setGate((current) => {
      if (current) {
        setAcknowledged({ topic: current, at: Date.now() });
        logEvent('GUIDANCE', `acknowledged — ${current.replace(/_/g, ' ')}`);
      }
      return null;
    });
  }, []);

  const [view, setView] = useState<ConsoleView>(
    () => viewFromHash(window.location.hash) ?? 'monitoring',
  );

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash(window.location.hash) ?? 'monitoring');
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const selectView = (next: ConsoleView) => {
    setView(next);
    window.location.hash = `/${next}`;
  };
  const alarmCount = useAlarmCount();

  // Staleness is a function of wall-clock time, not of any arriving frame, so
  // it needs its own tick. Without it a link that goes quiet leaves the last
  // frame on screen looking perfectly current.
  const now = useNow(250);
  const stale = isStale(telemetry.timestamp, now);
  const available = telemetry.connected && !stale;
  // `now` is sampled on its own 250 ms tick, so a frame that arrived between
  // ticks is briefly "in the future". Clamp rather than show a negative age.
  const ageMs = telemetry.timestamp === 0 ? null : Math.max(0, now - telemetry.timestamp);

  const blocked = gate !== null;

  return (
    <div className={styles.console}>
      {/* The console stays visible and keeps updating behind the gate — the
          checklist is meant to be read against the live state it refers to.
          `inert` removes it from the tab order and from pointer events, so a
          keystroke or click cannot reach it while the gate is open. */}
      <div className={styles.stack} {...(blocked ? { inert: '' } : {})}>
          <StatusStrip
          link={link}
          linkUp={telemetry.connected}
          stale={stale}
          sensorSource={wrench?.source}
          robotState={telemetry.robotState}
          frameRateHz={frameRateHz}
          paused={paused}
          onTogglePause={() => setPaused(!paused)}
          onReset={resetSession}
        />

        {paused ? (
          <div className={styles.hold} role="status">
            <span className="tag tag--strong">HOLD</span>
            <span>
              Display frozen. Telemetry is still being received and the stage classifier is
              still running; only the display is held.
            </span>
          </div>
        ) : null}

        <div className={styles.main}>
          <NavRail view={view} onSelect={selectView} alarmCount={alarmCount} />

          <div className={styles.workspace}>
            <div className={styles.viewport}>
              <Workspace
                jointPositions={telemetry.jointPositions}
                available={available}
                contactPhase={contact.phase}
                trajectory={trajectory}
              />
            </div>
            <div
              className={`${styles.lower} ${view === 'monitoring' ? styles.lowerChart : ''}`}
            >
              {view === 'monitoring' ? <ForceTrend history={history} /> : null}
              {view === 'teleoperation' ? (
                <JointRates telemetry={telemetry} available={available} />
              ) : null}
              {view === 'contact' ? <ModeTimeline history={history} contact={contact} /> : null}
              {view === 'safety' ? (
                <SafetyPanel telemetry={telemetry} contact={contact} available={available} />
              ) : null}
            </div>
          </div>

          <StatusColumn
            telemetry={telemetry}
            wrench={wrench}
            contact={contact}
            peakN={peak}
            available={available}
            acknowledged={acknowledged}
            frameCount={history.length}
          />
        </div>

          <EventLog ageMs={ageMs} stale={stale} />
      </div>

      {gate ? <GuidanceGate topic={gate} onAcknowledge={acknowledge} /> : null}
    </div>
  );
}

function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

/** Warn and alarm events in the log, shown against the SAFETY nav item. */
function useAlarmCount(): number {
  const [count, setCount] = useState(0);
  useEffect(
    () =>
      subscribeEvents((events) =>
        setCount(events.filter((e) => e.severity !== 'info').length),
      ),
    [],
  );
  return count;
}
