import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import { Fr5Model } from './Fr5Model';
import { STACK } from './ProbeAssembly';
import { usePalette } from '../telemetry/theme';
import styles from './Workspace.module.css';

interface Props {
  jointPositions?: number[];
  available: boolean;
  contactPhase: 'approach' | 'contact';
  /** Draw the probe and sensor frames. The views that are about frames ask
   *  for them; the rest do not, and three triads at the tool crowd the view. */
  showFrames?: boolean;
  /** Recent flange positions, oldest first, in metres. */
  trajectory: [number, number, number][];
}

/**
 * Robot workspace.
 *
 * Drawn as an engineering view on white: grey ground rule, dark grey links,
 * navy joints, black flange. The arm comes from forward kinematics on the
 * incoming joint angles rather than from a mesh, so nothing on screen can hold
 * a pose the telemetry did not supply.
 *
 * Colour carries no state here. Contact is shown by a black ring and a printed
 * label; a joint approaching its limit gets a black band and appears in the
 * joint table as text.
 */
export function Workspace({
  jointPositions,
  available,
  contactPhase,
  trajectory,
  showFrames,
}: Props) {
  return (
    <section className="plate">
      <div className="plate__head">
        <span className="plate__title">Robot workspace</span>
        <span className="plate__aside">
          FR5 · forward kinematics · drag to orbit
        </span>
      </div>
      <div className={`plate__body ${styles.body}`}>
        {!available ? (
          <div className={styles.veil}>
            <span className="tag tag--strong">NO TELEMETRY</span>
          </div>
        ) : null}

        <Canvas
          camera={{ position: [0.86, 0.62, 0.86], fov: 40 }}
          dpr={[1, 2]}
          gl={{ antialias: true }}
          shadows="soft"
          style={{ background: '#ffffff' }}
        >
          {/* Lit so the links have volume, not so the render looks expensive.
              One key with a soft shadow, a fill from the opposite side to keep
              the shaded faces readable, and a low ambient floor. The earlier
              flat 2.1 ambient washed every surface to the same value, which is
              what made the arm read as an outline drawing. */}
          <hemisphereLight args={['#ffffff', '#e6ece7', 0.85]} />
          <ambientLight intensity={0.55} />
          <directionalLight
            position={[2.2, 3.4, 1.8]}
            intensity={1.35}
            castShadow
            shadow-mapSize={[2048, 2048]}
            shadow-camera-left={-1.2}
            shadow-camera-right={1.2}
            shadow-camera-top={1.2}
            shadow-camera-bottom={-1.2}
            shadow-bias={-0.0006}
          />
          <directionalLight position={[-2.4, 1.6, -1.9]} intensity={0.45} />
          <directionalLight position={[0, 0.6, -2.6]} intensity={0.22} />

          <GroundPlan />
          <ShadowCatcher />

          {/* URDF frames are Z-up; three.js is Y-up. */}
          <group rotation={[-Math.PI / 2, 0, 0]}>
            <Fr5Model
              jointPositions={jointPositions}
              available={available}
              contact={contactPhase === 'contact'}
              showFrames={showFrames}
            />
            <Trajectory points={trajectory} />
            {/* The scanning surface sits under wherever the probe currently is,
                at table height. A phantom parked at a fixed spot reads as
                scenery the arm happens to be near; one under the probe reads as
                the thing being scanned. */}
            <Phantom under={trajectory[trajectory.length - 1]} />
          </group>

          <OrbitControls
            enablePan={false}
            minDistance={0.28}
            maxDistance={3}
            maxPolarAngle={Math.PI / 2.05}
            target={[0, 0.42, 0]}
            makeDefault
          />
        </Canvas>

        <ScaleBar contact={contactPhase === 'contact'} available={available} />
      </div>
    </section>
  );
}

/** Square grid at 100 mm pitch, drawn flat so it reads as a plan, not a floor. */
/**
 * The surface the shadow lands on.
 *
 * A plain white plane rather than a visible floor: it takes the key light's
 * shadow and nothing else, so the arm sits on something without the viewport
 * gaining a stage. Slightly below the grid so the rules stay crisp on top.
 */
function ShadowCatcher() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
      <planeGeometry args={[3.2, 3.2]} />
      <shadowMaterial opacity={0.14} />
    </mesh>
  );
}

/**
 * Scanning surface under the probe.
 *
 * A matte light-grey curved section standing in for a phantom or a body
 * surface. It is here to give the contact geometry somewhere to happen — a
 * probe pressing against nothing shows the operator a force with no visible
 * cause.
 */
function Phantom({ under }: { under?: [number, number, number] }) {
  // Robot frame is Z-up here: x and y follow the probe, z is the table.
  const x = under ? under[0] : 0.45;
  const y = under ? under[1] : 0.0;
  return (
    <group position={[x, y, 0.018]}>
      <mesh receiveShadow scale={[1, 1, 0.42]}>
        <sphereGeometry args={[0.105, 40, 22, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#dcdedb" roughness={0.94} metalness={0} />
      </mesh>
      <PhantomRing />
    </group>
  );
}

/* Split out so the ring can take the palette without making `Phantom` a
   consumer — it has no other colour of its own. */
function PhantomRing() {
  const palette = usePalette();
  return (
    <mesh position={[0, 0, -0.017]}>
      <ringGeometry args={[0.104, 0.110, 48]} />
      <meshBasicMaterial color={palette['--rule']} />
    </mesh>
  );
}

function GroundPlan() {
  const palette = usePalette();
  const lines = useMemo(() => {
    const out: [number, number, number][][] = [];
    const half = 0.6;
    for (let i = -6; i <= 6; i += 1) {
      const at = i * 0.1;
      out.push([
        [at, 0, -half],
        [at, 0, half],
      ]);
      out.push([
        [-half, 0, at],
        [half, 0, at],
      ]);
    }
    return out;
  }, []);

  return (
    <group>
      {lines.map((points, i) => (
        <Line
          key={i}
          points={points}
          color={palette['--rule']}
          lineWidth={i % 2 === 0 ? 1 : 0.6}
        />
      ))}
    </group>
  );
}

/** Flange path over the recent past, drawn in the console's structural accent. */
function Trajectory({ points }: { points: [number, number, number][] }) {
  const palette = usePalette();
  if (points.length < 2) return null;
  return <Line points={points} color={palette['--green']} lineWidth={1.2} dashed={false} />;
}

function ScaleBar({ contact, available }: { contact: boolean; available: boolean }) {
  return (
    <div className={styles.scaleBar}>
      <span className={styles.scaleItem}>Grid 100 mm</span>
      <span className={styles.scaleItem}>TCP frame: long = X · mid = Y · short = Z</span>
      <span className={styles.scaleItem}>Path 10 s</span>
      <span className={styles.scaleSpacer} />
      {available ? (
        <span className={contact ? 'tag tag--strong' : 'tag tag--off'}>
          {contact ? 'Contact indicated' : 'No contact'}
        </span>
      ) : null}
      {/* The tool is real now — 234 mm of adapter, sensor, bracket and probe,
          the bracket and probe drawn from their own CAD — so the caption says
          what is drawn rather than what is missing. It reads the stack rather
          than repeating it, so it cannot drift from what the scene shows. */}
      <span className={styles.caveat}>
        Tool {Math.round(STACK.total * 1000)} mm · sensor{' '}
        {Math.round(STACK.mountFace * 1000)} mm
      </span>
    </div>
  );
}
