import { Euler, Matrix4, Quaternion, Vector3 } from 'three';

/**
 * FR5 kinematic chain, taken from Fairino's own `fairino_description` URDF
 * (`fairino5_v6.urdf`) rather than from a datasheet reading.
 *
 * Each joint carries a fixed transform from its parent frame followed by a
 * rotation about local Z. This is the URDF convention, and using it directly
 * means the rendered arm and the meshes share one definition — the meshes are
 * authored in these same joint frames, with `visual origin = 0 0 0`.
 *
 * The values reproduce the FR5's published 922 mm reach:
 *   0.425 + 0.395 + 0.102 = 0.922 m.
 *
 * The earlier Denavit-Hartenberg table agreed on lengths but could not place
 * the meshes, because DH frames are not the frames the CAD is drawn in.
 */
export interface UrdfJoint {
  name: string;
  /** Fixed translation from the parent joint frame, metres. */
  xyz: [number, number, number];
  /** Fixed rotation from the parent joint frame, fixed-axis XYZ, radians. */
  rpy: [number, number, number];
  /** Mesh drawn in this joint's frame. */
  mesh: string;
}

export const FR5_CHAIN: UrdfJoint[] = [
  { name: 'j1', xyz: [0, 0, 0], rpy: [0, 0, 0], mesh: 'shoulder_link' },
  { name: 'j2', xyz: [0, 0, 0.152], rpy: [Math.PI / 2, 0, 0], mesh: 'upperarm_link' },
  { name: 'j3', xyz: [-0.425, 0, 0], rpy: [0, 0, 0], mesh: 'forearm_link' },
  { name: 'j4', xyz: [-0.39501, 0, 0], rpy: [0, 0, 0], mesh: 'wrist1_link' },
  { name: 'j5', xyz: [0, 0, 0.1021], rpy: [Math.PI / 2, 0, 0], mesh: 'wrist2_link' },
  { name: 'j6', xyz: [0, 0, 0.102], rpy: [-Math.PI / 2, 0, 0], mesh: 'wrist3_link' },
];

export const BASE_MESH = 'base_link';

/** probe.yaml `safety.joint_limits_deg`. */
export const JOINT_LIMITS_DEG = {
  lower: [-175, -265, -160, -265, -175, -165],
  upper: [175, 85, 160, 85, 175, 165],
};

export const JOINT_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'];

export interface LinkPose {
  position: Vector3;
  quaternion: Quaternion;
}

/**
 * Cumulative pose of every joint frame, plus the flange.
 *
 * Returns `FR5_CHAIN.length + 1` entries: index i is the frame of joint i+1
 * (so the mesh for `FR5_CHAIN[i]` is drawn there), and the final entry is the
 * tool flange at the end of the last link.
 */
export function solveChain(jointPositions: number[] | undefined): LinkPose[] {
  const running = new Matrix4().identity();
  const out: LinkPose[] = [];
  const scratchQuat = new Quaternion();

  for (let i = 0; i < FR5_CHAIN.length; i += 1) {
    const joint = FR5_CHAIN[i];
    running.multiply(
      new Matrix4().compose(
        new Vector3(...joint.xyz),
        scratchQuat.setFromEuler(new Euler(joint.rpy[0], joint.rpy[1], joint.rpy[2], 'XYZ')),
        new Vector3(1, 1, 1),
      ),
    );
    running.multiply(new Matrix4().makeRotationZ(jointPositions?.[i] ?? 0));
    out.push(decompose(running));
  }

  // The flange sits at the origin of the last joint frame; wrist3's own mesh
  // already carries the mounting face, so no extra offset is applied.
  out.push(decompose(running));
  return out;
}

function decompose(matrix: Matrix4): LinkPose {
  const position = new Vector3();
  const quaternion = new Quaternion();
  const scale = new Vector3();
  matrix.clone().decompose(position, quaternion, scale);
  return { position, quaternion };
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
