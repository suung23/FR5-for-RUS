import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import { Fr5Model } from './Fr5Model';
import styles from './Workspace.module.css';

interface Props {
  jointPositions?: number[];
  available: boolean;
  contactPhase: 'approach' | 'contact';
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
export function Workspace({ jointPositions, available, contactPhase, trajectory }: Props) {
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
          camera={{ position: [1.05, 0.8, 1.05], fov: 44 }}
          dpr={[1, 2]}
          gl={{ antialias: true }}
          style={{ background: '#ffffff' }}
        >
          {/* Flat, even lighting. A technical drawing does not want a key
              light — specular streaks would read as state changes. */}
          <ambientLight intensity={2.1} />
          <directionalLight position={[2.5, 4, 2]} intensity={0.9} />
          <directionalLight position={[-2, 1.5, -2]} intensity={0.5} />

          <GroundPlan />

          {/* URDF frames are Z-up; three.js is Y-up. */}
          <group rotation={[-Math.PI / 2, 0, 0]}>
            <Fr5Model
              jointPositions={jointPositions}
              available={available}
              contact={contactPhase === 'contact'}
            />
            <Trajectory points={trajectory} />
          </group>

          <OrbitControls
            enablePan={false}
            minDistance={0.5}
            maxDistance={3}
            maxPolarAngle={Math.PI / 2.05}
            target={[0, 0.34, 0]}
            makeDefault
          />
        </Canvas>

        <ScaleBar contact={contactPhase === 'contact'} available={available} />
      </div>
    </section>
  );
}

/** Square grid at 100 mm pitch, drawn flat so it reads as a plan, not a floor. */
function GroundPlan() {
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
          color="#d6d9de"
          lineWidth={i % 2 === 0 ? 1 : 0.6}
        />
      ))}
    </group>
  );
}

/** Flange path over the recent past, drawn as a thin navy polyline. */
function Trajectory({ points }: { points: [number, number, number][] }) {
  if (points.length < 2) return null;
  return <Line points={points} color="#10233f" lineWidth={1.2} dashed={false} />;
}

function ScaleBar({ contact, available }: { contact: boolean; available: boolean }) {
  return (
    <div className={styles.scaleBar}>
      <span className={styles.scaleItem}>GRID 100 mm</span>
      <span className={styles.scaleItem}>TCP FRAME: LONG=X · MID=Y · NAVY=Z</span>
      <span className={styles.scaleItem}>PATH 10 s</span>
      <span className={styles.scaleSpacer} />
      {available ? (
        <span className={contact ? 'tag tag--strong' : 'tag tag--off'}>
          {contact ? 'CONTACT RING SHOWN' : 'NO CONTACT'}
        </span>
      ) : null}
      <span className={styles.caveat}>Flange shown — tool transform unmeasured</span>
    </div>
  );
}
