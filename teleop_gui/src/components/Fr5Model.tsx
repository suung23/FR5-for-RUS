import { Line } from '@react-three/drei';
import { useLoader } from '@react-three/fiber';
import { Suspense, useMemo } from 'react';
import { BufferGeometry } from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import {
  BASE_MESH,
  FR5_CHAIN,
  marginToLimit,
  solveChain,
  type LinkPose,
} from '../telemetry/fr5Model';
import { ProbeAssembly, STACK } from './ProbeAssembly';

const MESH_URL = (name: string) => `./models/fr5/${name}.stl`;

/** Painted link housings. */
const SHELL = '#f4f5f2';
/** Joint collars and interface hardware. */
const ORANGE = '#e86f24';
/** Recessed detail and mechanical edges. */
const GRAPHITE = '#4a4a46';
/** Deep green — reserved for state, never for structure. */
const ACTIVE = '#17683a';

interface Props {
  jointPositions?: number[];
  available: boolean;
  contact: boolean;
  /** Draw the probe and sensor coordinate frames. Off outside the views that
   *  are about frames, because three triads at the tool crowd the viewport. */
  showFrames?: boolean;
}

/**
 * The FR5, drawn from Fairino's own `fairino_description` meshes.
 *
 * Seven STL parts placed on the URDF chain: the arm on screen is the arm in
 * the cell, at the joint angles the bridge reported.
 *
 * Shaded as the machine rather than as a drawing. The earlier version traced
 * every sharp edge in near-black over flat grey faces, which read as a pencil
 * sketch — legible as a diagram, but it invited the operator to take the
 * viewport as a schematic. Painted housings, orange joint collars and lighting
 * with real falloff give each link volume, and volume is what tells you which
 * way the wrist is turned.
 *
 * Colour still means only one thing. The arm is the machine's own livery;
 * deep green appears on the tool frame, the flange path and contact, which are
 * *state*. A joint inside ten degrees of its stop darkens its collar, and the
 * joint table reports the same fact as a number, so nothing depends on a
 * colour being seen.
 */
export function Fr5Model({ jointPositions, available, contact, showFrames }: Props) {
  const poses = useMemo(() => solveChain(jointPositions), [jointPositions]);
  const flange = poses[poses.length - 1];

  return (
    <Suspense fallback={null}>
      <Part url={MESH_URL(BASE_MESH)} dimmed={!available} />
      {FR5_CHAIN.map((joint, i) => {
        const margin = jointPositions ? marginToLimit(i, jointPositions[i]) : Infinity;
        const nearLimit = margin < (10 * Math.PI) / 180;
        return (
          <group key={joint.name}>
            <Part url={MESH_URL(joint.mesh)} pose={poses[i]} dimmed={!available} />
            <JointCollar pose={poses[i]} dimmed={!available} nearLimit={nearLimit} />
          </group>
        );
      })}
      <group
        position={flange.position.toArray() as [number, number, number]}
        quaternion={flange.quaternion.toArray() as [number, number, number, number]}
      >
        <ProbeAssembly dimmed={!available} contact={contact} showFrames={showFrames} />
        {showFrames && available ? <ToolFrame /> : null}
        {contact && available ? <ContactMark /> : null}
      </group>
    </Suspense>
  );
}

function Part({
  url,
  pose,
  dimmed,
}: {
  url: string;
  pose?: LinkPose;
  dimmed: boolean;
}) {
  const geometry = useLoader(STLLoader, url) as BufferGeometry;

  const position = pose ? pose.position.toArray() : [0, 0, 0];
  const quaternion = pose
    ? (pose.quaternion.toArray() as [number, number, number, number])
    : ([0, 0, 0, 1] as [number, number, number, number]);

  return (
    <mesh
      geometry={geometry}
      position={position as [number, number, number]}
      quaternion={quaternion}
      castShadow
      receiveShadow
    >
      {/* Painted metal: matte, barely metallic. A glossier surface would throw
          highlights that move with the camera and read as parts of the shape. */}
      <meshStandardMaterial
        color={dimmed ? '#fbfbfa' : SHELL}
        roughness={0.62}
        metalness={0.06}
      />
    </mesh>
  );
}

/**
 * The orange band at a joint.
 *
 * The supplied meshes are one solid per link, so a joint cover cannot be given
 * its own material by splitting the geometry. A collar drawn on the joint axis
 * puts the colour where the machine wears it and, more usefully, marks the
 * axis itself — the thing an operator is actually looking for when reading a
 * pose off the screen.
 */
function JointCollar({
  pose,
  dimmed,
  nearLimit,
}: {
  pose: LinkPose;
  dimmed: boolean;
  nearLimit: boolean;
}) {
  return (
    <group
      position={pose.position.toArray() as [number, number, number]}
      quaternion={pose.quaternion.toArray() as [number, number, number, number]}
    >
      {/* Cylinders are +Y in three.js; the joint axis is local +Z. */}
      <mesh rotation={[Math.PI / 2, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.0455, 0.0455, 0.026, 40]} />
        <meshStandardMaterial
          color={dimmed ? '#e8e8e6' : nearLimit ? GRAPHITE : ORANGE}
          roughness={0.55}
          metalness={0.08}
        />
      </mesh>
      {/* A seam either side, so the collar reads as a fitted cover rather than
          a painted stripe. */}
      {[0.0145, -0.0145].map((y) => (
        <mesh key={y} position={[0, 0, y]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.0462, 0.0462, 0.0022, 40]} />
          <meshStandardMaterial color={GRAPHITE} roughness={0.7} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Probe control frame at the acoustic reference point.
 *
 * Axes are told apart by length rather than the usual red/green/blue triad,
 * which this palette does not allow: long is +x (lateral, in the image plane),
 * middle is +y (elevational), short is +z (axial — the direction the probe
 * presses).
 */
function ToolFrame() {
  return (
    <group position={[0, 0, STACK.total]}>
      <Axis to={[0.075, 0, 0]} width={2.0} />
      <Axis to={[0, 0.055, 0]} width={1.5} />
      <Axis to={[0, 0, 0.038]} width={1.5} />
    </group>
  );
}

function ContactMark() {
  return (
    <group position={[0, 0, STACK.total]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.03, 0.038, 32]} />
        <meshBasicMaterial color={ACTIVE} />
      </mesh>
      {/* Normal-force arrow, along the axis the controller regulates. */}
      <Line
        points={[
          [0, 0, 0],
          [0, 0, 0.052],
        ]}
        color={ACTIVE}
        lineWidth={2.4}
      />
    </group>
  );
}

function Axis({ to, width }: { to: [number, number, number]; width: number }) {
  return (
    <Line
      points={[
        [0, 0, 0],
        to,
      ]}
      color={ACTIVE}
      lineWidth={width}
    />
  );
}
