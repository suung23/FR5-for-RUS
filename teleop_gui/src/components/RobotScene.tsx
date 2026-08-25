import { Canvas } from '@react-three/fiber';
import { Grid, OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import { DoubleSide, Quaternion, Vector3 } from 'three';
import {
  forwardKinematics,
  marginToLimit,
  type FramePose,
} from '../telemetry/kinematics';
import styles from './RobotScene.module.css';

interface Props {
  jointPositions?: number[];
  available: boolean;
  contactPhase: 'approach' | 'contact';
}

/**
 * Live pose view.
 *
 * The arm is drawn from the joint angles through forward kinematics rather
 * than from an imported mesh. That keeps the picture honest: every segment on
 * screen is a consequence of a number that arrived over the wire, and nothing
 * is drawn from a model file that could stay in a stale pose if telemetry
 * stopped.
 *
 * Styling is deliberately schematic — capsule links and ring joints — because
 * a photorealistic arm would imply a fidelity the kinematic constants do not
 * have. The tool transform is still unmeasured, so the tip marker is labelled
 * as the flange, not the probe contact point.
 */
export function RobotScene({ jointPositions, available, contactPhase }: Props) {
  const poses = useMemo(() => forwardKinematics(jointPositions), [jointPositions]);

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="label">Pose · forward kinematics from joint angles</span>
        <span className={styles.hint}>drag to orbit · scroll to zoom</span>
      </div>
      <div className={`panel__body panel__body--flush ${styles.body}`}>
        {!available ? <div className={styles.veil}>Telemetry unavailable</div> : null}
        <Canvas
          camera={{ position: [1.35, 1.05, 1.35], fov: 42 }}
          dpr={[1, 2]}
          gl={{ antialias: true }}
          style={{ background: 'transparent' }}
        >
          <hemisphereLight args={['#9fc4e0', '#0b0f14', 0.85]} />
          <directionalLight position={[3, 5, 2]} intensity={1.15} />
          <directionalLight position={[-3, 2, -2]} intensity={0.3} color="#7cc4e8" />

          <Grid
            args={[4, 4]}
            cellSize={0.1}
            cellThickness={0.5}
            cellColor="#1c2531"
            sectionSize={0.5}
            sectionThickness={0.9}
            sectionColor="#2b3646"
            fadeDistance={5}
            fadeStrength={1.4}
            infiniteGrid
            position={[0, 0, 0]}
          />

          <Pedestal />
          <Arm poses={poses} dimmed={!available} jointPositions={jointPositions} />
          <Flange pose={poses[poses.length - 1]} contact={contactPhase === 'contact'} dimmed={!available} />

          <OrbitControls
            enablePan={false}
            minDistance={0.7}
            maxDistance={4}
            maxPolarAngle={Math.PI / 2.05}
            target={[0, 0.35, 0]}
            makeDefault
          />
        </Canvas>

        <Legend />
      </div>
    </section>
  );
}

function Pedestal() {
  return (
    <group>
      <mesh position={[0, 0.012, 0]}>
        <cylinderGeometry args={[0.115, 0.135, 0.024, 48]} />
        <meshStandardMaterial color="#232c38" roughness={0.85} metalness={0.15} />
      </mesh>
      {/* A faint ground disc anchors the arm without the infinite grid reading
          as a floor the robot is floating above. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <ringGeometry args={[0.135, 0.32, 64]} />
        <meshBasicMaterial color="#141b24" side={DoubleSide} transparent opacity={0.75} />
      </mesh>
    </group>
  );
}

function Arm({
  poses,
  dimmed,
  jointPositions,
}: {
  poses: FramePose[];
  dimmed: boolean;
  jointPositions?: number[];
}) {
  return (
    <group>
      {poses.slice(0, -1).map((from, i) => {
        const to = poses[i + 1];
        const margin = jointPositions ? marginToLimit(i, jointPositions[i]) : Infinity;
        // Joints inside 10 degrees of a limit are tinted. This is the one place
        // colour appears on the arm, so it cannot be mistaken for shading.
        const nearLimit = margin < (10 * Math.PI) / 180;
        return (
          <group key={i}>
            <Segment from={from.position} to={to.position} dimmed={dimmed} />
            <JointRing pose={from} dimmed={dimmed} nearLimit={nearLimit} />
          </group>
        );
      })}
    </group>
  );
}

function Segment({
  from,
  to,
  dimmed,
}: {
  from: Vector3;
  to: Vector3;
  dimmed: boolean;
}) {
  const { position, quaternion, length } = useMemo(() => {
    const delta = new Vector3().subVectors(to, from);
    const len = delta.length();
    const mid = new Vector3().addVectors(from, to).multiplyScalar(0.5);
    const q = new Quaternion().setFromUnitVectors(
      new Vector3(0, 1, 0),
      len > 1e-6 ? delta.clone().normalize() : new Vector3(0, 1, 0),
    );
    return { position: mid, quaternion: q, length: len };
  }, [from, to]);

  if (length < 1e-4) return null;

  return (
    <mesh position={position} quaternion={quaternion}>
      <capsuleGeometry args={[0.035, Math.max(0.001, length - 0.07), 6, 20]} />
      <meshStandardMaterial
        color={dimmed ? '#2a323d' : '#4d5c6f'}
        roughness={0.55}
        metalness={0.35}
      />
    </mesh>
  );
}

function JointRing({
  pose,
  dimmed,
  nearLimit,
}: {
  pose: FramePose;
  dimmed: boolean;
  nearLimit: boolean;
}) {
  return (
    <mesh position={pose.position} quaternion={pose.quaternion}>
      <torusGeometry args={[0.05, 0.014, 12, 28]} />
      <meshStandardMaterial
        color={dimmed ? '#333c48' : nearLimit ? '#e0a44a' : '#7cc4e8'}
        emissive={nearLimit && !dimmed ? '#5c3d12' : '#0d1a22'}
        roughness={0.4}
        metalness={0.4}
      />
    </mesh>
  );
}

function Flange({
  pose,
  contact,
  dimmed,
}: {
  pose: FramePose;
  contact: boolean;
  dimmed: boolean;
}) {
  return (
    <group position={pose.position} quaternion={pose.quaternion}>
      <mesh>
        <cylinderGeometry args={[0.042, 0.042, 0.016, 28]} />
        <meshStandardMaterial color={dimmed ? '#3a434f' : '#93a6bd'} roughness={0.4} metalness={0.5} />
      </mesh>
      {/* Contact halo: the only element that changes with force state, so the
          3D view carries the same fact as the mode panel without a second
          reading of the number. */}
      {contact && !dimmed ? (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.055, 0.075, 32]} />
          <meshBasicMaterial color="#e0a44a" side={DoubleSide} transparent opacity={0.7} />
        </mesh>
      ) : null}
    </group>
  );
}

function Legend() {
  return (
    <div className={styles.legend}>
      <span>
        <i className={styles.swatchJoint} /> joint
      </span>
      <span>
        <i className={styles.swatchLimit} /> within 10° of limit
      </span>
      <span>
        <i className={styles.swatchFlange} /> J6 flange
      </span>
      <span className={styles.caveat}>
        Tool transform unmeasured — flange shown, not probe tip
      </span>
    </div>
  );
}
