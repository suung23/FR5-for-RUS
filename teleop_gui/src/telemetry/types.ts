/**
 * Telemetry contract shared by every transport.
 *
 * Everything except `timestamp` and `connected` is optional on purpose. A
 * transport reports only the fields its source actually carries, and the UI
 * renders "—" for the rest. That distinction matters here: a joint angle of 0
 * and a joint angle we were never told are different facts, and collapsing
 * them would let the 3D view show a confident pose built from nothing.
 */

export type RobotState = 'idle' | 'teleop' | 'contact' | 'fault' | 'estop';

/**
 * Velocity-limit mode declared by the control stack.
 *
 * Distinct from the console's own contact classification: this is what the
 * robot is *actually* clamping to, reported by `us_diff_ik_node`. The console
 * must never infer it — if the display guessed and the robot disagreed, the
 * operator would be reading a limit that is not in force.
 */
export type ProbingMode = 'approach' | 'contact_probing';

export type SafetyState =
  | 'normal'
  | 'warning'
  | 'protective_stop'
  | 'emergency_stop';

export interface TcpPose {
  /** Metres, robot base frame. */
  position: [number, number, number];
  /** Unit quaternion, `[x, y, z, w]`. */
  quaternion: [number, number, number, number];
}

export interface RobotTelemetry {
  /** Epoch milliseconds, stamped by the transport on receipt. */
  timestamp: number;
  connected: boolean;
  /** Radians, 6-DOF, base to flange. */
  jointPositions?: number[];
  /** Radians per second. */
  jointVelocities?: number[];
  tcpPose?: TcpPose;
  robotState?: RobotState;
  safetyState?: SafetyState;
  /** Velocity-limit mode the control stack has entered. */
  probingMode?: ProbingMode;
}

/**
 * Force/torque sample in the probe frame.
 *
 * The PX6D reaches the workstation over USB, not through the robot
 * controller, so it arrives on its own path and carries its own clock. Keeping
 * it in a separate record rather than folding it into `RobotTelemetry` keeps
 * that fact visible instead of implying a single synchronised source.
 */
export type WrenchSource = 'px6d_serial' | 'controller' | 'simulation' | 'none';

export interface WrenchSample {
  timestamp: number;
  /** Newtons, `[Fx, Fy, Fz]`. */
  force: [number, number, number];
  /** Newton-metres, `[Mx, My, Mz]`. */
  torque: [number, number, number];
  /**
   * Where the numbers came from.
   *
   * The panel used to be labelled "PX6D" unconditionally, which was untrue
   * whenever the wrench arrived on the controller topic — and on this cell it
   * usually did, because the PX6D is the USB variant and never reaches the
   * controller at all. A force reading whose origin is guessed is worse than
   * no reading.
   */
  source?: WrenchSource;
  /** CRC mismatches since the stream synchronised. Direct serial only. */
  crcErrors?: number;
  /** Frames per second measured at the sensor. Direct serial only. */
  sensorHz?: number;
}

export type TransportKind = 'websocket' | 'http' | 'rosbridge' | 'simulation';

export type LinkPhase =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error'
  | 'closed';

export interface LinkStatus {
  phase: LinkPhase;
  transport: TransportKind;
  /** Human-readable endpoint, shown in the header so the operator can see
   *  at a glance whether this is live hardware or the built-in simulator. */
  endpoint: string;
  /** Last transport-level error, if any. Never a thrown exception — the UI
   *  must keep rendering the last known state when a link drops. */
  error?: string;
  /** Epoch ms of the most recent frame from this transport. */
  lastFrameAt?: number;
  /** Consecutive failed connection attempts. */
  attempts: number;
}

/** Callbacks a transport uses to push into the adapter. */
export interface TransportSink {
  onTelemetry(frame: RobotTelemetry): void;
  onWrench(sample: WrenchSample): void;
  onStatus(patch: Partial<LinkStatus>): void;
}

/**
 * A transport owns one connection and nothing else. It does no smoothing, no
 * unit conversion beyond parsing, and no policy — those live above it so that
 * swapping websocket for rosbridge cannot change what the numbers mean.
 */
export interface Transport {
  readonly kind: TransportKind;
  readonly endpoint: string;
  start(sink: TransportSink): void;
  stop(): void;
}
