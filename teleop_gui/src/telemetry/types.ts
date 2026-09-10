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
/**
 * Velocity-limit stage, as `us_diff_ik` declares it.
 *
 * `contact_probing_inplane` is contact probing with one axis handed back: the
 * robot still holds the force on `z`, and the operator gets `ω_y` — rotation
 * about the elevational axis, the only rotation that maps the imaging plane
 * (the probe's x–z plane) onto itself. Everything else stays at zero, so the
 * beam keeps sweeping the same plane in space while the operator rocks in it.
 */
export type ProbingMode =
  | 'approach'
  | 'contact_probing'
  | 'contact_probing_inplane'
  /** Contact regime held by the policy runner rather than by the force judgement.
   *  Keeps the `contact_probing` prefix on purpose: every consumer tests that prefix
   *  (us_servo_node keeps small commands alive, run_policy reads the handover). */
  | 'contact_probing_policy';

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
  /**
   * Whether the in-plane rotation mode is selected, as `us_diff_ik` holds it.
   *
   * Separate from `probingMode` on purpose: the mode string is what the
   * velocity limits *are*, and this is what they will be on contact. Selecting
   * it during approach is the normal way to use it — contact starts without
   * warning, and at that moment the operator's hand is on the stylus.
   */
  inplaneRotation?: boolean;
  /** How the control stack is mapping the operator's hand onto the robot. */
  teleopFrame?: TeleopFrameState;
  /** Sensor calibration state, forwarded by the bridge. */
  calibration?: CalibrationStatus;
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
  /** How many sensor samples this frame is the mean of. 0 = not averaged. */
  averagedSamples?: number;
  /**
   * Per-axis force `[min, max]` over the same window the mean covers.
   *
   * The displayed value is an average; a spike inside the window does not
   * survive it. These are what the peak-hold reads, so a transient the
   * operator needs to know about is not lost to smoothing.
   */
  forceExtremes?: [number[], number[]];
  /**
   * The shape of the window this frame summarises, folded into bins.
   *
   * The bridge refreshes the readout once a second because that is the rate a
   * number on a screen can be read at. A line is not a number: at one point per
   * second a 20 s trend is a twenty-step staircase, which is a 1 kHz sensor
   * drawn as if it were a 1 Hz one. This block carries the window's own shape,
   * so the readout stays calm while the plot shows what the sensor saw.
   *
   * Frames arrive faster than the readout changes (`bridge.wrench_stream_hz`,
   * 10 Hz) and each carries only the bins closed since the last one — the
   * readout fields simply repeat in between. Resolution and arrival rate are
   * separate: a second's worth of bins delivered once a second would draw at
   * full 100 Hz resolution and still lurch forward in one-second slabs, which
   * is a trace that is stamped rather than streamed.
   *
   * Each bin is a mean *and* a min/max, not a chosen sample. Picking one
   * sample per bin would alias; folding the bin cannot.
   */
  forceWaveform?: ForceWaveform;
  /** Every stage of the calibration pipeline, when a profile is loaded. */
  compensated?: CompensatedStages;
  calibrationValid?: boolean;
  calibrationIssues?: string[];
}

/** A window of force, binned for drawing. Bins are chronological. */
export interface ForceWaveform {
  /** Bins per second. 100 Hz means each bin covers 10 ms. */
  hz: number;
  bins: ForceWaveformBin[];
}

export interface ForceWaveformBin {
  /** Milliseconds before the frame that carried it. Never negative. */
  ageMs: number;
  /** Bin mean, newtons. */
  fx: number;
  fy: number;
  fz: number;
  /** Extremes of `F_z` inside the bin — the excursion the mean would hide. */
  fzMin: number;
  fzMax: number;
}

/**
 * The teleoperation mapping, as `us_diff_ik` declares it.
 *
 * Read, never recomputed. Which way a hand motion is read is a property of the
 * control stack; a console that worked it out for itself could disagree with
 * the robot about which way "right" is, and the operator would have no way to
 * tell which of the two was lying.
 */
export interface TeleopFrameState {
  /**
   * Where the operator is standing, as a rotation about the base vertical.
   *
   * 0 = alongside the robot, facing the same way. 180 = facing it, which is
   * what "mirror" means here: fore/aft and left/right both flip. Not a
   * reflection — walking around to the other side flips both, and a reflection
   * would break rotation commands, which are pseudovectors.
   */
  operatorYawDeg: number;
  /** Requested but not yet in effect. Applies at the next grip, not mid-motion. */
  pendingYawDeg: number | null;
  mirrored: boolean;
  /** `latched` removes drift; `body` is the older behaviour, kept for comparison. */
  linearFrame: string;
  angularFrame: string;
  tipRollDeg: number;
  /** Grips since the node started. The latch is taken on the first one. */
  engageCount: number;
}

/**
 * The compensation pipeline, stage by stage.
 *
 * Intermediate stages are kept because the verification page has to show
 * "the raw value is this, and after compensation it is that" side by side —
 * which stage a reading goes wrong at is which calibration is wrong.
 */
export interface CompensatedStages {
  rawSensor: number[];
  biasCorrectedSensor: number[];
  externalSensor: number[];
  externalProbe: number[];
  contactProbe: number[];
  normalForceN: number;
}

/** Progress of a calibration session in the bridge. */
export interface CalibrationSession {
  capturing: string;
  captureLabel: string;
  captureProgress: number;
  captureSamples: number;
  biasAccepted: boolean | null;
  biasReason: string;
  poseCount: number;
  poseLabels: string[];
  poseTarget: number;
  coverage: number;
  lastError: string;
  /** Which direction of pose is short, said as an instruction. */
  coverageHint?: string;
  /**
   * The model the operator has just fitted, before any save.
   *
   * Typed out rather than left as an opaque record: the calibration page reads
   * these fields to show what `Fit model` produced. Without that the button
   * changed nothing on screen, and the operator had to decide whether to save
   * a result they could not see.
   *
   * Snake-case because it is `GravityModel.to_dict()` passed through verbatim.
   */
  gravity: GravityFit | null;
}

/** `GravityModel.to_dict()` from the bridge, field names unchanged. */
export interface GravityFit {
  mass_kg: number;
  com_sensor_m: number[];
  residual_bias: number[];
  rms_force_n: number;
  rms_torque_nm: number;
  per_axis_force_n: number[];
  per_axis_torque_nm: number[];
  coverage: number;
  poses: number;
  valid: boolean;
  issues: string[];
}

/** Calibration state as the bridge reports it. */
export interface CalibrationStatus {
  path: string;
  /** Residual removed at the working pose, probe frame. Null when not taken. */
  workingTare?: number[] | null;
  /** How far the arm is now from the pose the tare was taken at, degrees. */
  tareOffAxisDeg?: number | null;
  /** Auto-capture is armed: a drag that settles records a pose by itself. */
  autoCapture?: boolean;
  /** Movement has been seen and the next settle will be taken. */
  autoArmed?: boolean;
  present: boolean;
  valid: boolean;
  issues: string[];
  mountingAngleDeg: number;
  axialFlip: boolean;
  leverSensorToProbeM: number[];
  rotationProbeFromSensor: number[][];
  session: CalibrationSession;
  createdAt?: number;
  ageHours?: number;
  mountingNote?: string;
  biasAccepted?: boolean;
  biasReason?: string;
  bias?: number[];
  massKg?: number | null;
  comSensorM?: number[] | null;
  rmsForceN?: number | null;
  rmsTorqueNm?: number | null;
  perAxisForceN?: number[] | null;
  perAxisTorqueNm?: number[] | null;
  coverage?: number | null;
  poses?: number | null;
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
/**
 * The bridge's answer to a console command.
 *
 * Every calibration command can be refused, and the refusal carries the only
 * explanation the operator will get — the probe is not pointing down, there is
 * load on it, the pose has not arrived. Dropping these on the floor is what
 * makes a button look dead when it is in fact being turned away.
 */
export interface CommandAck {
  command: string;
  ok: boolean;
  reason?: string;
}

/**
 * One ultrasound picture, already fan-converted and JPEG-encoded by the bridge.
 *
 * The bridge is not the probe's client — `us_frame_node` is, because the probe
 * accepts only one. This is a display copy; what a session records is always the
 * raw polar frame.
 */
export interface UltrasoundFrame {
  /** Bridge clock, ms. */
  timestamp: number;
  /** Monotonic counter from the bridge. Lets the view tell a stalled stream from a still image. */
  seq: number;
  width: number;
  height: number;
  /** `fan` (scan-converted) or `polar` (raw layout). */
  display: string;
  /** base64 JPEG, no data: prefix. */
  jpeg: string;
  /** When the renderer received it. */
  receivedAt: number;
}

/**
 * The perception verdict for one frame, as `run_segmentation` reports it.
 *
 * Read, never recomputed. Every number here comes from `ControlState`
 * (`Unet_seg/rus_perception`), which is the same record the policy's
 * observation vector is built from — so what the panel prints is what the
 * network actually said, not a second opinion derived from the picture.
 *
 * `quality` is `Q_seg`, and it **presupposes the bladder was found**. When
 * `hasMask` is false the score is not a low score, it is not a score; the panel
 * has to say that rather than print a number the operator will read as "poor
 * image".
 */
export interface SegmentationState {
  seq?: number;
  checkpointId?: string;
  /** `Q_seg`, the control quality score. Null when it was not measured. */
  quality: number | null;
  /** The observation verdict. **Not** a command and not a claim about the arm. */
  validForControl: boolean;
  /** Machine-readable reasons the observation was rejected. Empty when valid. */
  rejectionReasons: string[];
  hasMask: boolean;
  maskAreaPx?: number;
  /** Mask area over the ROI — the imaged sector, not the whole frame. */
  maskAreaRatio: number | null;
  /** Lumen centroid in pixels of the 256² B-mode, `[x, y]`. */
  centroidPx: [number, number] | null;
  /** Normalized centroid − 0.5. `+x` right, `+y` down. */
  centerError: [number, number] | null;
  /** Horizontal centre of the imaged sector. `eHat` is measured from this, not from the frame centre. */
  beamAxisPx: number | null;
  /** `beamAxis − centroid_x`, pixels. Positive = lumen left of the beam axis. */
  eHatPx: number | null;
  /** Discrete action token: `check-filling` | `hold` | `move-left` | `move-right`. */
  token: string | null;
  segmentationConfidence: number | null;
  lumenContrast: number | null;
  borderContactRatio: number | null;
  largestComponentRatio: number | null;
  temporalWarpedIou: number | null;
  centroidJump: number | null;
  maskThreshold: number | null;
  roiMode?: string;
  /** Wall-clock cost of one U-Net pass, measured inside the node. */
  perceptionMs: number | null;
  /** Frames the node dropped to stay under its rate cap. */
  skipped?: number;
}

/**
 * One segmented frame: the picture the network saw, and its mask.
 *
 * The base image is **not** the sector shown in the Ultrasound panel. That one
 * is `fr5_vision.scan_convert`'s fan; this one is `rus_policy.bmode`'s 256²
 * letterbox, which is what the network is actually fed. Laying the mask over
 * the other picture would put the boundary in the wrong place and look
 * entirely convincing, so the two travel together and are paired by the ROS
 * header stamp in the bridge before either is sent.
 *
 * The mask arrives binary and uncoloured. Fill, outline and opacity are the
 * panel's decision, because the operator has to be able to blink the mask off
 * and check the boundary against the lumen edge underneath it.
 */
export interface SegmentationFrame {
  /** Bridge clock, ms. */
  timestamp: number;
  /** Monotonic counter from the bridge. Distinguishes a still image from a dead stream. */
  seq: number;
  width: number;
  height: number;
  /** base64 JPEG of the B-mode the network saw, no data: prefix. */
  jpeg: string;
  /** base64 PNG of the binary mask, no data: prefix. PNG because JPEG ringing would fake a soft edge. */
  mask: string;
  /** The perception verdict, when the state message paired with this picture. */
  state?: SegmentationState;
  /** How far behind the picture the state was. A large value means they are not the same frame. */
  stateAgeMs?: number;
  /** When the renderer received it. */
  receivedAt: number;
}

export interface TransportSink {
  onTelemetry(frame: RobotTelemetry): void;
  onUltrasound(frame: UltrasoundFrame): void;
  onSegmentation(frame: SegmentationFrame): void;
  onWrench(sample: WrenchSample): void;
  onStatus(patch: Partial<LinkStatus>): void;
  onAck(ack: CommandAck): void;
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
  /**
   * Send a console command, if this transport has a back channel.
   *
   * **Never a motion command.** No transport accepts one and none ever will:
   * the single path that moves the robot stays with the control stack that
   * owns the velocity clamps and the watchdogs. What travels here is "capture
   * at the pose the operator has already moved to", "fit", "save", and which
   * side of the robot the operator is standing on — none of which move
   * anything by themselves.
   *
   * Returns false when the transport cannot send (no socket, or read-only).
   */
  sendCommand?(command: Record<string, unknown>): boolean;
}
