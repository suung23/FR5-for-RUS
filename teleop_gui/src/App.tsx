import { useCallback, useEffect, useState } from 'react';
import { EventLog } from './components/EventLog';
import { GuidanceGate } from './components/GuidanceGate';
import { ForceTrend } from './components/ForceTrend';
import { JointRates } from './components/JointRates';
import { ModeTimeline } from './components/ModeTimeline';
import { NavRail, viewFromHash, type ConsoleView } from './components/NavRail';
import { OperatorFrame } from './components/OperatorFrame';
import { ProbingNotice } from './components/ProbingNotice';
import { ProbingMode } from './components/ProbingMode';
import { SafetyPanel } from './components/SafetyPanel';
import { SensorCalibration } from './components/SensorCalibration';
import { StatusColumn } from './components/StatusColumn';
import { StatusStrip } from './components/StatusStrip';
import { Ultrasound } from './components/Ultrasound';
import { Workspace } from './components/Workspace';
import { isStale } from './store/selectors';
import { logEvent } from './store/events';
import { subscribeEvents } from './store/events';
import { refreshPalette, setProbingTheme } from './telemetry/theme';
import { useTelemetryStore } from './store/telemetryStore';
import styles from './App.module.css';

export function App() {
  const telemetry = useTelemetryStore((s) => s.telemetry);
  const wrench = useTelemetryStore((s) => s.wrench);
  const ultrasound = useTelemetryStore((s) => s.ultrasound);
  const contact = useTelemetryStore((s) => s.contact);
  const contactForce = useTelemetryStore((s) => s.contactForce);
  const contactJudged = useTelemetryStore((s) => s.contactJudged);
  const link = useTelemetryStore((s) => s.link);
  const history = useTelemetryStore((s) => s.history);
  const waveformHz = useTelemetryStore((s) => s.waveformHz);
  const trajectory = useTelemetryStore((s) => s.trajectory);
  const peak = useTelemetryStore((s) => s.peakContactForceN);
  const frameRateHz = useTelemetryStore((s) => s.frameRateHz);
  const paused = useTelemetryStore((s) => s.paused);
  const setPaused = useTelemetryStore((s) => s.setPaused);
  const resetSession = useTelemetryStore((s) => s.resetSession);
  const sendCommand = useTelemetryStore((s) => s.sendCommand);

  // The selected view lives in the URL hash. A reload — or a crash and restart
  // mid-procedure — returns to the panel the operator was on rather than to a
  // default they then have to re-select.
  // Operator briefing. Shown once at startup and never again — the console
  // stays inert until it is acknowledged. It does not re-arm on mode changes:
  // an overlay that reappears mid-procedure covers the live state at the moment
  // the operator is reacting to it, and gets dismissed unread.
  const [gate, setGate] = useState(true);
  const [acknowledged, setAcknowledged] = useState<{ at: number } | null>(null);

  const acknowledge = useCallback(() => {
    setGate((open) => {
      if (open) {
        setAcknowledged({ at: Date.now() });
        logEvent('GUIDANCE', 'operator briefing acknowledged');
      }
      return false;
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

  const blocked = gate;

  // Contact probing repaints the whole console blue and raises a standing
  // notice. Both are driven from the mode the **control stack declares**, never
  // from the console's own contact classification — the two can disagree, and
  // if they do, what the operator needs to see is what the arm is actually
  // clamping to.
  //
  // Gated on `available` so a dead link cannot leave the console sitting in the
  // probing palette on the strength of a frame that stopped arriving minutes
  // ago. When the link drops, the colour goes back to resting and the status
  // strip says the link is stale — which is the true statement of what is
  // known.
  const probing = available && telemetry.probingMode === 'contact_probing';

  useEffect(() => {
    setProbingTheme(probing);
  }, [probing]);

  // On unmount put the palette back. The attribute lives on the document, not
  // in React's tree, so nothing else would. The mount side re-resolves the
  // tokens now that the document is fully styled.
  useEffect(() => {
    refreshPalette();
    return () => setProbingTheme(false);
  }, []);

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

        {probing ? <ProbingNotice /> : null}

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
              {/* The picture leads. It is what the operator is actually reading
                  while probing; the arm beside it is context. On the frame views
                  it stands down — the triads need that width. */}
              {view === 'monitoring' || view === 'contact' ? (
                <Ultrasound frame={ultrasound} className={styles.usPanel} />
              ) : null}
              <Workspace
                jointPositions={telemetry.jointPositions}
                available={available}
                contactPhase={contact.phase}
                trajectory={trajectory}
                // Frames belong to the views that are about frames. On the
                // monitoring view they would sit on top of the force reading
                // the operator is there to watch.
                showFrames={view === 'calibration' || view === 'contact'}
              />
            </div>
            <div
              className={`${styles.lower} ${
              view === 'monitoring' ? styles.lowerChart : ''
            } ${view === 'calibration' || view === 'safety' ? styles.lowerTall : ''}`}
            >
              {view === 'monitoring' ? (
                <ForceTrend
                  history={history}
                  waveformHz={waveformHz}
                />
              ) : null}
              {view === 'teleoperation' ? (
                <>
                  <OperatorFrame
                    telemetry={telemetry}
                    available={available}
                    onCommand={sendCommand}
                  />
                  <JointRates telemetry={telemetry} available={available} />
                </>
              ) : null}
              {view === 'contact' ? (
                <>
                  <ModeTimeline history={history} contact={contact} />
                  <ProbingMode
                    telemetry={telemetry}
                    available={available}
                    onCommand={sendCommand}
                  />
                </>
              ) : null}
              {view === 'calibration' ? (
              <SensorCalibration
                telemetry={telemetry}
                wrench={wrench}
                available={available}
                onCommand={sendCommand}
              />
            ) : null}
            {view === 'safety' ? (
                <SafetyPanel telemetry={telemetry} contact={contact} available={available} />
              ) : null}
            </div>
          </div>

          <StatusColumn
            telemetry={telemetry}
            wrench={wrench}
            contact={contact}
            contactForce={contactForce}
            contactJudged={contactJudged}
            peakN={peak}
            available={available}
            acknowledged={acknowledged}
            sendCommand={sendCommand}
          />
        </div>

          <EventLog ageMs={ageMs} stale={stale} />
      </div>

      {gate ? <GuidanceGate onAcknowledge={acknowledge} /> : null}
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
