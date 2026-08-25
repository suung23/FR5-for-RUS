import { Line } from '@react-three/drei';
import { useLoader } from '@react-three/fiber';
import { Suspense, useMemo } from 'react';
import { BufferGeometry, EdgesGeometry } from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import {
  BASE_MESH,
  FR5_CHAIN,
  marginToLimit,
  solveChain,
  type LinkPose,
} from '../telemetry/fr5Model';

const MESH_URL = (name: string) => `./models/fr5/${name}.stl`;

/** Sharp-edge threshold. Below this the STL's tessellation shows as noise. */
const EDGE_ANGLE_DEG = 32;

interface Props {
  jointPositions?: number[];
  available: boolean;
  contact: boolean;
}

/**
 * The real FR5, drawn from Fairino's own `fairino_description` meshes.
 *
 * Seven STL parts placed on the URDF chain: the arm on screen is the arm in
 * the cell, at the joint angles the bridge reported. The earlier capsule
 * stand-in was honest about the joint angles but not about the machine, and a
 * console that shows a schematic invites the operator to read it as one.
 *
 * Rendered as a technical drawing rather than a product shot — light grey
 * surfaces with black sharp edges. That keeps the strict console palette,
 * separates the arm from the white plate behind it, and makes the form legible
 * without relying on specular highlights the flat lighting does not provide.
 */
export function Fr5Model({ jointPositions, available, contact }: Props) {
  const poses = useMemo(() => solveChain(jointPositions), [jointPositions]);

  return (
    <Suspense fallback={null}>
      <Part url={MESH_URL(BASE_MESH)} dimmed={!available} />
      {FR5_CHAIN.map((joint, i) => {
        const margin = jointPositions ? marginToLimit(i, jointPositions[i]) : Infinity;
        return (
          <group key={joint.name}>
            <Part
              url={MESH_URL(joint.mesh)}
              pose={poses[i]}
              dimmed={!available}
              nearLimit={margin < (10 * Math.PI) / 180}
            />
          </group>
        );
      })}
      <Flange pose={poses[poses.length - 1]} contact={contact} dimmed={!available} />
    </Suspense>
  );
}

function Part({
  url,
  pose,
  dimmed,
  nearLimit,
}: {
  url: string;
  pose?: LinkPose;
  dimmed: boolean;
  nearLimit?: boolean;
}) {
  const geometry = useLoader(STLLoader, url) as BufferGeometry;
  const edges = useMemo(
    () => new EdgesGeometry(geometry, EDGE_ANGLE_DEG),
    [geometry],
  );

  const position = pose ? pose.position.toArray() : [0, 0, 0];
  const quaternion = pose
    ? (pose.quaternion.toArray() as [number, number, number, number])
    : ([0, 0, 0, 1] as [number, number, number, number]);

  return (
    <group position={position as [number, number, number]} quaternion={quaternion}>
      <mesh geometry={geometry} castShadow={false} receiveShadow={false}>
        <meshStandardMaterial
          color={dimmed ? '#ffffff' : '#d6d9de'}
          roughness={0.85}
          metalness={0.05}
          // Polygon offset keeps the edge lines from z-fighting with the faces
          // they trace, which at this line width reads as a dashed outline.
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial
          color={nearLimit && !dimmed ? '#000000' : dimmed ? '#d6d9de' : '#6b7280'}
          transparent
          opacity={nearLimit && !dimmed ? 1 : 0.85}
        />
      </lineSegments>
    </group>
  );
}

/**
 * Tool flange marker.
 *
 * Not a mesh — the probe mount is not modelled and `tool.j6_to_probe` is still
 * unmeasured, so this marks the J6 mounting face and nothing beyond it. The
 * TCP frame is three black rules of different length, since the palette has no
 * room for the usual red/green/blue triad.
 */
function Flange({
  pose,
  contact,
  dimmed,
}: {
  pose: LinkPose;
  contact: boolean;
  dimmed: boolean;
}) {
  if (dimmed) return null;
  return (
    <group
      position={pose.position.toArray() as [number, number, number]}
      quaternion={pose.quaternion.toArray() as [number, number, number, number]}
    >
      <TcpAxis to={[0.09, 0, 0]} width={1.6} />
      <TcpAxis to={[0, 0.065, 0]} width={1.2} />
      <TcpAxis to={[0, 0, 0.045]} width={1.2} navy />
      {contact ? (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.046, 0.058, 28]} />
          <meshBasicMaterial color="#000000" />
        </mesh>
      ) : null}
    </group>
  );
}

function TcpAxis({
  to,
  width,
  navy,
}: {
  to: [number, number, number];
  width: number;
  navy?: boolean;
}) {
  return (
    <Line
      points={[
        [0, 0, 0],
        to,
      ]}
      color={navy ? '#10233f' : '#000000'}
      lineWidth={width}
    />
  );
}
