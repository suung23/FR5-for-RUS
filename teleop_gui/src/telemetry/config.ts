import type { TransportKind } from './types';

/**
 * Environment-driven configuration.
 *
 * The default transport is the **telemetry bridge** — `ros2 run fr5_control
 * telemetry_bridge` on this workstation, serving ws://localhost:8765. The
 * console is a monitor for a real session, so its resting state has to be
 * "waiting for the robot", never a robot that moves on its own.
 *
 * The FR5 controller itself speaks only XML-RPC/UDP and serves no websocket,
 * which is what the bridge exists to solve: it subscribes to the Phase 0
 * stack's own topics and republishes them in this contract.
 *
 * `simulation` is still available but must be asked for explicitly. It exists
 * for reviewing the interface without hardware, and every panel labels it.
 *
 * This application is receive-only. No transport here has a send path, and the
 * adapter exposes none. Motion commands stay with the ROS stack that owns the
 * safety limits.
 */
export interface AppConfig {
  robotHost: string;
  transport: TransportKind;
  telemetryUrl: string;
  wrenchUrl?: string;
  /** Poll period for the `http` transport, milliseconds. */
  httpPollMs: number;
  /** rosbridge topic names. */
  rosJointTopic: string;
  rosWrenchTopic: string;
  /** Contact detection, mirroring `probe.yaml` safety limits. */
  contactEnterN: number;
  contactReleaseN: number;
  warnForceN: number;
  maxForceN: number;
  /** Force at which the control stack drops to contact-probing limits. */
  contactProbingN: number;
  /** Force the robot holds once in contact probing, and the band around it. */
  targetForceN: number;
  targetBandN: number;
  normalForceSign: number;
}

function str(key: string, fallback: string): string {
  const raw = import.meta.env[key as keyof ImportMetaEnv];
  return typeof raw === 'string' && raw.length > 0 ? raw : fallback;
}

function num(key: string, fallback: number): number {
  // `Number('')` is 0, not NaN, so an unset variable would otherwise resolve to
  // zero rather than to the fallback — and a zero force threshold reads as a
  // configured value rather than a missing one.
  const raw = str(key, '');
  if (raw.trim() === '') return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function transportKind(): TransportKind {
  const raw = str('VITE_ROBOT_TELEMETRY_TRANSPORT', 'websocket').toLowerCase();
  if (
    raw === 'websocket' ||
    raw === 'http' ||
    raw === 'rosbridge' ||
    raw === 'simulation'
  ) {
    return raw;
  }
  // An unrecognised value must not silently start generating data — that is
  // exactly the failure where an operator watches a robot that is not there.
  console.warn(
    `Unknown VITE_ROBOT_TELEMETRY_TRANSPORT "${raw}" — falling back to websocket.`,
  );
  return 'websocket';
}

export function loadConfig(): AppConfig {
  const robotHost = str('VITE_ROBOT_HOST', '192.168.58.3');
  const transport = transportKind();

  // The bridge runs on the workstation that has the ROS stack, not on the
  // robot. Pointing the default at the robot's own IP would look reasonable
  // and never connect.
  const impliedUrl =
    transport === 'http'
      ? `http://localhost:8080/telemetry`
      : transport === 'rosbridge'
        ? `ws://localhost:9090`
        : `ws://localhost:8765`;

  return {
    robotHost,
    transport,
    telemetryUrl: str('VITE_ROBOT_TELEMETRY_URL', impliedUrl),
    wrenchUrl: str('VITE_WRENCH_URL', ''),
    httpPollMs: num('VITE_HTTP_POLL_MS', 100),
    rosJointTopic: str('VITE_ROS_JOINT_TOPIC', '/fr5_right/joint_states'),
    rosWrenchTopic: str('VITE_ROS_WRENCH_TOPIC', '/fr5_right/wrench'),

    // probe.yaml: safety.max_normal_force_n = 10.0, warn_normal_force_n = 9.0,
    // teleop.contact_probing_force_n = 8.0, watchdog.retreat_until_force_n = 0.2,
    // ft_sensor.normal_force_sign = -1.0.
    contactEnterN: num('VITE_CONTACT_ENTER_N', 7.0),
    contactReleaseN: num('VITE_CONTACT_RELEASE_N', 0.2),
    warnForceN: num('VITE_WARN_FORCE_N', 9.0),
    maxForceN: num('VITE_MAX_FORCE_N', 10.0),
    contactProbingN: num('VITE_CONTACT_PROBING_N', 8.0),
    targetForceN: num('VITE_TARGET_FORCE_N', 5.0),
    targetBandN: num('VITE_TARGET_BAND_N', 0.5),
    normalForceSign: num('VITE_NORMAL_FORCE_SIGN', -1.0),
  };
}

export const config = loadConfig();

/** True when nothing is being asked of the lab network. */
export const isSimulated = config.transport === 'simulation';
