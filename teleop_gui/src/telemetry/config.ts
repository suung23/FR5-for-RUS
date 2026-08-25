import type { TransportKind } from './types';

/**
 * Environment-driven configuration.
 *
 * The default transport is `simulation`, deliberately. The robot at
 * 192.168.58.3 is a Fairino FR5 whose controller speaks its own XML-RPC/UDP
 * protocol; it does not serve a websocket or a ROS bridge unless something on
 * this network was set up to publish one. Defaulting to a live transport would
 * mean opening sockets against lab hardware on the strength of a guess, so a
 * real link has to be asked for explicitly through the environment.
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
  normalForceSign: number;
}

function str(key: string, fallback: string): string {
  const raw = import.meta.env[key as keyof ImportMetaEnv];
  return typeof raw === 'string' && raw.length > 0 ? raw : fallback;
}

function num(key: string, fallback: number): number {
  const parsed = Number(str(key, ''));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function transportKind(): TransportKind {
  const raw = str('VITE_ROBOT_TELEMETRY_TRANSPORT', 'simulation').toLowerCase();
  if (
    raw === 'websocket' ||
    raw === 'http' ||
    raw === 'rosbridge' ||
    raw === 'simulation'
  ) {
    return raw;
  }
  // An unrecognised value must not silently fall through to a live socket.
  console.warn(
    `Unknown VITE_ROBOT_TELEMETRY_TRANSPORT "${raw}" — falling back to simulation.`,
  );
  return 'simulation';
}

export function loadConfig(): AppConfig {
  const robotHost = str('VITE_ROBOT_HOST', '192.168.58.3');
  const transport = transportKind();

  // Only used when the operator picked a live transport without naming a URL.
  const impliedUrl =
    transport === 'http'
      ? `http://${robotHost}:8080/telemetry`
      : `ws://${robotHost}:9090`;

  return {
    robotHost,
    transport,
    telemetryUrl: str('VITE_ROBOT_TELEMETRY_URL', impliedUrl),
    wrenchUrl: str('VITE_WRENCH_URL', ''),
    httpPollMs: num('VITE_HTTP_POLL_MS', 100),
    rosJointTopic: str('VITE_ROS_JOINT_TOPIC', '/fr5_right/joint_states'),
    rosWrenchTopic: str('VITE_ROS_WRENCH_TOPIC', '/fr5_right/wrench'),

    // probe.yaml: safety.max_normal_force_n = 7.0, warn_normal_force_n = 6.0,
    // watchdog.retreat_until_force_n = 0.2, ft_sensor.normal_force_sign = -1.0.
    contactEnterN: num('VITE_CONTACT_ENTER_N', 7.0),
    contactReleaseN: num('VITE_CONTACT_RELEASE_N', 0.2),
    warnForceN: num('VITE_WARN_FORCE_N', 6.0),
    maxForceN: num('VITE_MAX_FORCE_N', 7.0),
    normalForceSign: num('VITE_NORMAL_FORCE_SIGN', -1.0),
  };
}

export const config = loadConfig();

/** True when nothing is being asked of the lab network. */
export const isSimulated = config.transport === 'simulation';
