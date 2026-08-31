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
  /**
   * Contact detection, mirroring `probe.yaml`'s mode switch.
   *
   * These are compared against the **contact-force magnitude**, not the normal
   * component — `ft_sensor.contact_force_mode: magnitude`.
   */
  contactEnterN: number;
  contactReleaseN: number;
  warnForceN: number;
  maxForceN: number;
  /** Force at which the control stack drops to contact-probing limits. */
  contactProbingN: number;
  /**
   * Force below which it returns to approach, and how long it must stay there.
   *
   * The transition became reversible on 2026-08-31, when the entry threshold
   * came down to 1 N. The console states the way back because the operator has
   * no other way to learn it: in contact probing their five non-force axes are
   * commanded to zero, so nothing they do with the stylus demonstrates it.
   */
  contactProbingReleaseN: number;
  contactProbingReleaseS: number;
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

    // probe.yaml: safety.max_normal_force_n = 15.0, warn_normal_force_n = 14.0,
    // teleop.contact_probing_force_n = 1.0, contact_probing_release_n = 0.3,
    // contact_control.target_force_n = 3.0, deadband_n = 0.5,
    // watchdog.retreat_until_force_n = 0.2, ft_sensor.normal_force_sign = -1.0.
    //
    // The 15 N limit is a temporary value with no clinical basis — see the
    // history in probe.yaml. It must not be read here as a reviewed number.
    //
    // ⚠️ These mirror probe.yaml by hand, and nothing enforces that. The three
    // that the gauge *draws* — probing, target, band — are the ones that lie
    // visibly when they drift: a probing mark at 8 N while the arm switches at
    // 1 N puts the operator's whole sense of margin in the wrong place. They
    // were last matched on 2026-08-31, when the contact threshold moved from
    // `F_n ≥ 8 N` to `‖F‖ ≥ 1 N` and the hold became 3.0 ± 0.5 N of ‖F‖.
    // Enter is the same yaml key the gauge's probing mark draws
    // (teleop.contact_probing_force_n) — the console must not claim contact at
    // a different force from the one that changed the arm's speed limit.
    // Release is the mode switch's own exit, not the watchdog's retreat
    // threshold: the switch is what decides whether the console says CONTACT.
    contactEnterN: num('VITE_CONTACT_ENTER_N', 2.0),
    contactReleaseN: num('VITE_CONTACT_RELEASE_N', 0.3),
    warnForceN: num('VITE_WARN_FORCE_N', 14.0),
    maxForceN: num('VITE_MAX_FORCE_N', 15.0),
    contactProbingN: num('VITE_CONTACT_PROBING_N', 2.0),
    contactProbingReleaseN: num('VITE_CONTACT_PROBING_RELEASE_N', 0.3),
    contactProbingReleaseS: num('VITE_CONTACT_PROBING_RELEASE_S', 0.5),
    targetForceN: num('VITE_TARGET_FORCE_N', 3.0),
    targetBandN: num('VITE_TARGET_BAND_N', 0.5),
    normalForceSign: num('VITE_NORMAL_FORCE_SIGN', -1.0),
  };
}

export const config = loadConfig();

/** True when nothing is being asked of the lab network. */
export const isSimulated = config.transport === 'simulation';
