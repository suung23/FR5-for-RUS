import type {
  RobotState,
  RobotTelemetry,
  SafetyState,
  TcpPose,
  WrenchSample,
  WrenchSource,
} from '../types';

/**
 * Tolerant parsing of whatever a telemetry endpoint actually sends.
 *
 * Every field is validated before it reaches the store. A malformed frame
 * yields absent fields rather than `NaN` or `0`, because a zero that looks like
 * a reading is worse on this screen than a blank.
 */

const ROBOT_STATES: RobotState[] = ['idle', 'teleop', 'contact', 'fault', 'estop'];
const SAFETY_STATES: SafetyState[] = [
  'normal',
  'warning',
  'protective_stop',
  'emergency_stop',
];
const WRENCH_SOURCES: WrenchSource[] = ['px6d_serial', 'controller', 'simulation', 'none'];

function numberArray(value: unknown, length?: number): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out = value.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : NaN));
  if (out.some((v) => Number.isNaN(v))) return undefined;
  if (length !== undefined && out.length !== length) return undefined;
  return out;
}

function tcpPose(value: unknown): TcpPose | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Record<string, unknown>;
  const position = numberArray(raw.position, 3);
  const quaternion = numberArray(raw.quaternion, 4);
  if (!position || !quaternion) return undefined;
  return {
    position: position as [number, number, number],
    quaternion: quaternion as [number, number, number, number],
  };
}

function oneOf<T extends string>(value: unknown, allowed: T[]): T | undefined {
  return typeof value === 'string' && (allowed as string[]).includes(value)
    ? (value as T)
    : undefined;
}

export function parseTelemetry(raw: unknown, receivedAt: number): RobotTelemetry | null {
  if (!raw || typeof raw !== 'object') return null;
  const src = raw as Record<string, unknown>;

  // Prefer the source's own clock only if it looks like epoch milliseconds.
  // A ROS header in seconds, or a monotonic counter, would otherwise place the
  // sample decades away and silently break every age calculation on screen.
  const claimed = typeof src.timestamp === 'number' ? src.timestamp : undefined;
  const plausible =
    claimed !== undefined && Math.abs(claimed - receivedAt) < 60 * 60 * 1000;

  return {
    timestamp: plausible ? (claimed as number) : receivedAt,
    connected: src.connected !== false,
    jointPositions: numberArray(src.jointPositions ?? src.joint_positions ?? src.position),
    jointVelocities: numberArray(
      src.jointVelocities ?? src.joint_velocities ?? src.velocity,
    ),
    tcpPose: tcpPose(src.tcpPose ?? src.tcp_pose),
    robotState: oneOf(src.robotState ?? src.robot_state, ROBOT_STATES),
    safetyState: oneOf(src.safetyState ?? src.safety_state, SAFETY_STATES),
  };
}

export function parseWrench(raw: unknown, receivedAt: number): WrenchSample | null {
  if (!raw || typeof raw !== 'object') return null;
  const src = raw as Record<string, unknown>;

  // Accept both a flat `{force, torque}` shape and geometry_msgs/Wrench.
  const wrench = (src.wrench ?? src) as Record<string, unknown>;
  const force = vec3(wrench.force);
  const torque = vec3(wrench.torque);
  if (!force || !torque) return null;

  const sample: WrenchSample = { timestamp: receivedAt, force, torque };

  const source = oneOf(src.source, WRENCH_SOURCES);
  if (source) sample.source = source;
  if (typeof src.crcErrors === 'number' && Number.isFinite(src.crcErrors)) {
    sample.crcErrors = src.crcErrors;
  }
  if (typeof src.sensorHz === 'number' && Number.isFinite(src.sensorHz)) {
    sample.sensorHz = src.sensorHz;
  }
  return sample;
}

function vec3(value: unknown): [number, number, number] | undefined {
  const asArray = numberArray(value, 3);
  if (asArray) return asArray as [number, number, number];
  if (!value || typeof value !== 'object') return undefined;
  const o = value as Record<string, unknown>;
  const xyz = [o.x, o.y, o.z];
  if (xyz.every((v) => typeof v === 'number' && Number.isFinite(v))) {
    return xyz as [number, number, number];
  }
  return undefined;
}
