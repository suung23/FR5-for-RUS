import { useEffect, useMemo, useState } from 'react';
import { AppHeader } from './components/AppHeader';
import { ForceChart } from './components/ForceChart';
import { ForceGauge } from './components/ForceGauge';
import { GuidancePanel } from './components/GuidancePanel';
import { JointTable } from './components/JointTable';
import { ModePanel } from './components/ModePanel';
import { RobotScene } from './components/RobotScene';
import { isSimulated } from './telemetry/config';
import { buildGuidance } from './telemetry/guidance';
import { isStale } from './store/selectors';
import { useTelemetryStore } from './store/telemetryStore';
import styles from './App.module.css';

export function App() {
  const telemetry = useTelemetryStore((s) => s.telemetry);
  const wrench = useTelemetryStore((s) => s.wrench);
  const contact = useTelemetryStore((s) => s.contact);
  const link = useTelemetryStore((s) => s.link);
  const history = useTelemetryStore((s) => s.history);
  const peak = useTelemetryStore((s) => s.peakNormalForceN);
  const frameRateHz = useTelemetryStore((s) => s.frameRateHz);
  const paused = useTelemetryStore((s) => s.paused);
  const setPaused = useTelemetryStore((s) => s.setPaused);
  const resetSession = useTelemetryStore((s) => s.resetSession);

  // Staleness is a function of wall-clock time, not of any incoming frame, so
  // it needs its own tick. Without it a link that goes quiet leaves the last
  // frame on screen looking perfectly current.
  const now = useNow(250);
  const stale = isStale(telemetry.timestamp, now);
  const connected = telemetry.connected && !stale;

  const guidance = useMemo(
    () =>
      buildGuidance({
        telemetry,
        contact,
        connected: telemetry.connected,
        stale,
        simulated: isSimulated,
      }),
    [telemetry, contact, stale],
  );

  return (
    <div className={styles.shell}>
      <AppHeader
        link={link}
        connected={telemetry.connected}
        stale={stale}
        frameRateHz={frameRateHz}
        paused={paused}
        onTogglePause={() => setPaused(!paused)}
        onReset={resetSession}
      />

      {paused ? (
        <div className={styles.holdBar} role="status">
          Display held — incoming telemetry is still being received but is not shown.
        </div>
      ) : null}

      <main className={styles.grid}>
        <div className={styles.scene}>
          <RobotScene
            jointPositions={telemetry.jointPositions}
            available={connected}
            contactPhase={contact.phase}
          />
        </div>

        <div className={styles.side}>
          <ModePanel
            telemetry={telemetry}
            contact={contact}
            connected={telemetry.connected}
            stale={stale}
          />
          <ForceGauge
            normalForceN={contact.normalForceN}
            peakN={peak}
            available={connected && wrench !== null}
          />
          <GuidancePanel items={guidance} />
        </div>

        <div className={styles.chart}>
          <ForceChart
            history={history}
            wrench={connected ? (wrench?.force ?? null) : null}
            torque={connected ? (wrench?.torque ?? null) : null}
          />
        </div>

        <div className={styles.joints}>
          <JointTable
            jointPositions={telemetry.jointPositions}
            jointVelocities={telemetry.jointVelocities}
            available={connected}
          />
        </div>
      </main>
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
