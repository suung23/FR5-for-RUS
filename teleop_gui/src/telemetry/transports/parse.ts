import type {
  SegmentationFrame,
  SegmentationState,
  UltrasoundFrame,
  CommandAck,
  ForceWaveform,
  ForceWaveformBin,
  ProbingMode,
  RobotState,
  RobotTelemetry,
  SafetyState,
  TcpPose,
  TeleopFrameState,
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
// Every value the stack can declare must be here. One that is missing parses as
// `undefined` — "no mode declared" — so while the policy held the régime the
// console read it as the stack being down, never lit HANDOVER, and showed no
// change at all when Stop handed the stylus back.
const PROBING_MODES: ProbingMode[] = [
  'approach',
  'contact_probing',
  'contact_probing_inplane',
  'contact_probing_policy',
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
    probingMode: oneOf(src.probingMode ?? src.probing_mode, PROBING_MODES),
    inplaneRotation:
      typeof src.inplaneRotation === 'boolean' ? src.inplaneRotation : undefined,
    // Passed through as the control stack declared it, for the same reason as
    // the calibration block below.
    teleopFrame: teleopFrame(src.teleopFrame ?? src.teleop_frame),
    // Passed through as the bridge sent it. The calibration block is display
    // and gating state, not something the console recomputes — recomputing it
    // here would let the screen disagree with the control stack about whether
    // contact control may be enabled.
    calibration: (src.calibration as never) ?? undefined,
  };
}

/**
 * The teleoperation mapping, if the bridge sent one.
 *
 * `operatorYawDeg` is the one field the panel cannot do without — everything
 * else it shows is context. A frame missing it is dropped rather than rendered
 * as 0, which would read as "alongside" and could be exactly wrong.
 */
function teleopFrame(raw: unknown): TeleopFrameState | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const src = raw as Record<string, unknown>;
  if (typeof src.operatorYawDeg !== 'number' || !Number.isFinite(src.operatorYawDeg)) {
    return undefined;
  }
  const pending = src.pendingYawDeg;
  return {
    operatorYawDeg: src.operatorYawDeg,
    pendingYawDeg: typeof pending === 'number' && Number.isFinite(pending) ? pending : null,
    mirrored: src.mirrored === true,
    linearFrame: typeof src.linearFrame === 'string' ? src.linearFrame : '—',
    angularFrame: typeof src.angularFrame === 'string' ? src.angularFrame : '—',
    tipRollDeg: typeof src.tipRollDeg === 'number' ? src.tipRollDeg : 0,
    engageCount: typeof src.engageCount === 'number' ? src.engageCount : 0,
  };
}

/** The bridge's `{type:"ack"}` reply, or null when this is not one. */
export function parseAck(raw: unknown): CommandAck | null {
  if (!raw || typeof raw !== 'object') return null;
  const src = raw as Record<string, unknown>;
  if (src.type !== 'ack' || typeof src.command !== 'string') return null;
  return {
    command: src.command,
    ok: src.ok === true,
    reason: typeof src.reason === 'string' ? src.reason : undefined,
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
  if (typeof src.averagedSamples === 'number' && Number.isFinite(src.averagedSamples)) {
    sample.averagedSamples = src.averagedSamples;
  }
  if (Array.isArray(src.forceExtremes) && src.forceExtremes.length === 2) {
    const [lo, hi] = src.forceExtremes as unknown[];
    if (Array.isArray(lo) && Array.isArray(hi) && lo.length === 3 && hi.length === 3) {
      sample.forceExtremes = [lo as number[], hi as number[]];
    }
  }
  const waveform = forceWaveform(src.forceWaveform);
  if (waveform) sample.forceWaveform = waveform;
  if (src.compensated && typeof src.compensated === 'object') {
    sample.compensated = src.compensated as never;
  }
  if (typeof src.calibrationValid === 'boolean') {
    sample.calibrationValid = src.calibrationValid;
  }
  if (Array.isArray(src.calibrationIssues)) {
    sample.calibrationIssues = (src.calibrationIssues as unknown[]).map(String);
  }
  return sample;
}

/**
 * The binned window, if the bridge sent one.
 *
 * Bins arrive as flat arrays rather than objects — a hundred of them a second
 * is a hundred objects a second on the wire, and the field order is fixed by
 * the same contract that reads it here. A bin that does not parse is dropped
 * on its own; one bad bin must not cost the plot the other ninety-nine.
 */
function forceWaveform(raw: unknown): ForceWaveform | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const src = raw as Record<string, unknown>;
  if (typeof src.hz !== 'number' || !Number.isFinite(src.hz) || src.hz <= 0) return undefined;
  if (!Array.isArray(src.bins)) return undefined;

  const bins: ForceWaveformBin[] = [];
  for (const entry of src.bins) {
    const row = numberArray(entry, 6);
    if (!row) continue;
    bins.push({
      ageMs: Math.max(0, row[0]),
      fx: row[1],
      fy: row[2],
      fz: row[3],
      fzMin: row[4],
      fzMax: row[5],
    });
  }
  return bins.length > 0 ? { hz: src.hz, bins } : undefined;
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


/**
 * Ultrasound picture from the bridge.
 *
 * Rejects a frame with no payload rather than handing the view an empty `img`
 * src — a broken picture and "no signal" look the same on screen otherwise.
 */
/** A finite number, or null. A missing measurement must not arrive as 0. */
function orNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function pairOrNull(value: unknown): [number, number] | null {
  const pair = numberArray(value, 2);
  return pair ? (pair as [number, number]) : null;
}

function segmentationState(value: unknown): SegmentationState | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const src = value as Record<string, unknown>;
  const reasons = Array.isArray(src.rejectionReasons)
    ? src.rejectionReasons.filter((r): r is string => typeof r === 'string')
    : [];
  return {
    seq: orNull(src.seq) ?? undefined,
    checkpointId: typeof src.checkpointId === 'string' ? src.checkpointId : undefined,
    quality: orNull(src.quality),
    // Absent means "not told", and the honest reading of that is "not valid for
    // control" — never assume a verdict the node did not give.
    validForControl: src.validForControl === true,
    rejectionReasons: reasons,
    hasMask: src.hasMask === true,
    maskAreaPx: orNull(src.maskAreaPx) ?? undefined,
    maskAreaRatio: orNull(src.maskAreaRatio),
    centroidPx: pairOrNull(src.centroidPx),
    centerError: pairOrNull(src.centerError),
    beamAxisPx: orNull(src.beamAxisPx),
    eHatPx: orNull(src.eHatPx),
    token: typeof src.token === 'string' ? src.token : null,
    segmentationConfidence: orNull(src.segmentationConfidence),
    lumenContrast: orNull(src.lumenContrast),
    borderContactRatio: orNull(src.borderContactRatio),
    largestComponentRatio: orNull(src.largestComponentRatio),
    temporalWarpedIou: orNull(src.temporalWarpedIou),
    centroidJump: orNull(src.centroidJump),
    maskThreshold: orNull(src.maskThreshold),
    roiMode: typeof src.roiMode === 'string' ? src.roiMode : undefined,
    perceptionMs: orNull(src.perceptionMs),
    skipped: orNull(src.skipped) ?? undefined,
  };
}

/**
 * One segmented frame.
 *
 * Both images are required. A frame carrying the picture but not the mask is
 * dropped rather than shown, because a B-mode with no mask on it is
 * indistinguishable from a B-mode the network found no bladder in — and those
 * are opposite facts.
 */
export function parseSegmentation(
  raw: unknown,
  receivedAt: number,
): SegmentationFrame | null {
  if (!raw || typeof raw !== 'object') return null;
  const src = raw as Record<string, unknown>;
  const jpeg = src.jpeg;
  const mask = src.mask;
  if (typeof jpeg !== 'string' || jpeg.length === 0) return null;
  if (typeof mask !== 'string' || mask.length === 0) return null;
  return {
    timestamp: typeof src.timestamp === 'number' ? src.timestamp : receivedAt,
    seq: typeof src.seq === 'number' ? src.seq : 0,
    width: typeof src.width === 'number' ? src.width : 0,
    height: typeof src.height === 'number' ? src.height : 0,
    jpeg,
    mask,
    state: segmentationState(src.state),
    stateAgeMs: orNull(src.stateAgeMs) ?? undefined,
    receivedAt,
  };
}

export function parseUltrasound(raw: unknown, receivedAt: number): UltrasoundFrame | null {
  if (!raw || typeof raw !== 'object') return null;
  const src = raw as Record<string, unknown>;
  const jpeg = src.jpeg;
  if (typeof jpeg !== 'string' || jpeg.length === 0) return null;
  return {
    timestamp: typeof src.timestamp === 'number' ? src.timestamp : receivedAt,
    seq: typeof src.seq === 'number' ? src.seq : 0,
    width: typeof src.width === 'number' ? src.width : 0,
    height: typeof src.height === 'number' ? src.height : 0,
    display: typeof src.display === 'string' ? src.display : 'fan',
    jpeg,
    receivedAt,
  };
}
