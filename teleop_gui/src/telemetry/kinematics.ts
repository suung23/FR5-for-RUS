import { Matrix4, Quaternion, Vector3 } from 'three';

/**
 * Forward kinematics for the FR5, used only to draw the arm.
 *
 * These Denavit-Hartenberg values come from Fairino's published FR5
 * specification (922 mm reach, 5 kg payload), not from a measurement of this
 * particular cell, and the tool transform `j6_to_probe` is still `.nan` in
 * probe.yaml pending the mount CAD. So the rendered pose is a faithful picture
 * of the *joint angles* and an approximation of where the probe tip sits. The
 * UI labels the tip marker accordingly rather than presenting it as truth.
 */
export interface DhLink {
  /** Link length, metres. */
  a: number;
  /** Link offset, metres. */
  d: number;
  /** Link twist, radians. */
  alpha: number;
}

export const FR5_DH: DhLink[] = [
  { a: 0, d: 0.152, alpha: Math.PI / 2 },
  { a: -0.425, d: 0, alpha: 0 },
  { a: -0.395, d: 0, alpha: 0 },
  { a: 0, d: 0.102, alpha: Math.PI / 2 },
  { a: 0, d: 0.102, alpha: -Math.PI / 2 },
  { a: 0, d: 0.1, alpha: 0 },
];

/** probe.yaml `safety.joint_limits_deg`. */
export const JOINT_LIMITS_DEG = {
  lower: [-175, -265, -160, -265, -175, -165],
  upper: [175, 85, 160, 85, 175, 165],
};

export const JOINT_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'];

function dhMatrix(theta: number, link: DhLink): Matrix4 {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(link.alpha);
  const sa = Math.sin(link.alpha);

  // Standard (Denavit-Hartenberg) convention, column-major for three.js.
  return new Matrix4().set(
    ct, -st * ca, st * sa, link.a * ct,
    st, ct * ca, -ct * sa, link.a * st,
    0, sa, ca, link.d,
    0, 0, 0, 1,
  );
}

export interface FramePose {
  position: Vector3;
  quaternion: Quaternion;
  matrix: Matrix4;
}

/**
 * Cumulative frame poses, base first.
 *
 * Returns `links.length + 1` entries: index 0 is the base, index i+1 is the
 * frame after joint i. Callers draw a segment between consecutive entries.
 */
export function forwardKinematics(
  jointPositions: number[] | undefined,
  links: DhLink[] = FR5_DH,
): FramePose[] {
  const running = new Matrix4().identity();
  const poses: FramePose[] = [poseFrom(running)];

  for (let i = 0; i < links.length; i += 1) {
    const theta = jointPositions?.[i] ?? 0;
    running.multiply(dhMatrix(theta, links[i]));
    poses.push(poseFrom(running));
  }
  return poses;
}

function poseFrom(matrix: Matrix4): FramePose {
  const clone = matrix.clone();
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  clone.decompose(position, quaternion, scale);
  return { position, quaternion, matrix: clone };
}

/** Fraction of a joint's travel already used, 0 at the lower limit. */
export function jointTravelFraction(index: number, radians: number): number {
  const lo = (JOINT_LIMITS_DEG.lower[index] * Math.PI) / 180;
  const hi = (JOINT_LIMITS_DEG.upper[index] * Math.PI) / 180;
  if (!(hi > lo)) return 0.5;
  return Math.min(1, Math.max(0, (radians - lo) / (hi - lo)));
}

/** Radians remaining before the nearer limit of this joint. */
export function marginToLimit(index: number, radians: number): number {
  const lo = (JOINT_LIMITS_DEG.lower[index] * Math.PI) / 180;
  const hi = (JOINT_LIMITS_DEG.upper[index] * Math.PI) / 180;
  return Math.min(radians - lo, hi - radians);
}
