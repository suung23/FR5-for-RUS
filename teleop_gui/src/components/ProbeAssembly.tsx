import { Line } from '@react-three/drei';
import { useLoader } from '@react-three/fiber';
import { CatmullRomCurve3, Vector3, type BufferGeometry } from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { useMemo } from 'react';
import { usePalette } from '../telemetry/theme';

/**
 * The stack below the flange, in metres. **Mount v2 build, 2026-09-11.**
 *
 * Two sources, deliberately. `adapter` and `sensor` are measured — there is no
 * solid for either, so a tape is the only thing there is. `mount` is the CAD
 * (`hardware/probe_mount_v2`, 142.0 mm read off the solid), which supersedes
 * anything taped for it: where a solid exists, the solid wins.
 *
 * The two numbers that leave this file are `probe.yaml`'s, and they were
 * measured on the assembled arm rather than added up from here:
 *
 *   flange → sensor top   35.2 mm    `ft_sensor.j6_to_sensor_xyz`
 *   flange → array face  234.11 mm   `tool.j6_to_probe_xyz`
 *   ⇒ lever arm          198.91 mm   what the moment correction rides on
 *
 * `adapter` carries the 2.2 mm the stack grew over the v1 build (33.0 → 35.2).
 * It is put there rather than in `sensor` because the PX6D body is a fixed
 * 23 mm part and the adapter is the un-solid, measured spacer — but the split
 * is an attribution, not a measurement. Only the sum is measured.
 *
 * `probe` is what is left over (56.91 mm), and with v2 that number finally
 * means something: the v1 clamp was a smooth channel the probe slid in, so the
 * exposed length was whatever it happened to be. v2 grips the handle between
 * two flexure plates pulled by four M4 screws, so it is repeatable once tight.
 */
export const STACK = {
  /** Flange to the sensor's lower face. Measured — see the note above. */
  adapter: 0.0122,
  /** PX6D body. Measured. */
  sensor: 0.023,
  /** Bracket, from the v2 solid (142.0 mm). */
  mount: 0.142,
  /** Clamp face to the array face. The remainder of the measured 234.11 mm. */
  probe: 0.05691,
  get sensorFace() {
    return this.adapter;
  },
  get mountFace() {
    return this.adapter + this.sensor;
  },
  get probeFace() {
    return this.adapter + this.sensor + this.mount;
  },
  get total() {
    return this.adapter + this.sensor + this.mount + this.probe;
  },
};

const MESH_URL = (name: string) => `./models/probe/${name}.stl`;

/* Metalness stays low across the assembly. There is no environment map in
   this scene — only directional lights — and a physically metallic surface
   with nothing to reflect renders black. The parts are told apart by colour
   and roughness instead, which is what a matte bead-blasted finish looks like
   anyway. */

/** Machined aluminium, bead-blasted. */
const ALUMINIUM = '#b4b7b2';
/** Sensor body and hardware. */
const GRAPHITE = '#5c5f5a';
/** Probe housing. */
const PROBE_SHELL = '#d7d9d6';
/** Cable and strain relief. */
const CABLE = '#6e716d';

/**
 * Where the probe's image plane lies, in the flange frame.
 *
 * `probe.yaml` `tool.j6_to_probe_rpy` — the array's long axis sits 90° from
 * flange +x. Confirmed twice over: the operator measured it, and the teleop
 * `tip_roll_deg = 90` found by hand in 2026-08-19 turns out to be the same
 * rotation expressed in the other place (DESIGN_NOTES §10.4).
 */
const PROBE_YAW_DEG = 90;

/**
 * Where the PX6D's own channel axes lie, in the flange frame.
 *
 * `PROBE_YAW_DEG − 43°`, the 43 being the measured sensor-to-probe angle
 * (`px6d_axis_id`, `{P}R{S} = Rz(-θ)`). The triad is drawn on that diagonal
 * because the force numbers on screen are measured in it — an operator who
 * sees the sensor square to the screen will read the force axes off the
 * screen, and they are not the same axes.
 */
const SENSOR_YAW_DEG = PROBE_YAW_DEG - 43;

/**
 * The clamp's long axis in the bracket mesh's own frame — **0° for v2**.
 *
 * The v1 solid was drawn with the clamp at exactly 45.00°, so the console had
 * to take it back out. The v2 asset here is the *aligned* export
 * (`ultrasound_probe_mount_v2_aligned.stl`, converted mm → m), whose clamp
 * walls are already on the axes — so there is nothing to take out.
 *
 * The cross-check the 45° used to give is not lost, it just moved: the
 * as-built push test put the probe at 43° from the sensor's +x, and that 43°
 * still lives in `SENSOR_YAW_DEG` and in `probe.yaml`'s `j6_to_sensor_rpy`.
 */
const MOUNT_CLAMP_CAD_DEG = 0;

/**
 * The convex face, as fitted to the CAD (R = 82.6 mm, residual 0.14 mm).
 *
 * Used only to lay the contact indicator on the acoustic surface. A flat patch
 * there would sit inside the housing at the edges and float off it in the
 * middle.
 */
const ARRAY_RADIUS = 0.0826;
/** Half-angle subtended by the array footprint at that radius. */
const ARRAY_HALF_ANGLE = Math.asin(0.040 / ARRAY_RADIUS);
/** Elevational width of the head, from the CAD bounding box. */
const ARRAY_WIDTH = 0.031;

interface Props {
  dimmed: boolean;
  contact: boolean;
  showFrames?: boolean;
}

/**
 * Flange adapter → PX6D → mount → convex probe, in that order.
 *
 * Approximate but mechanically honest: each stage is a distinct part with its
 * own material and its own bolted interface, at the axial length it actually
 * occupies. What it is *not* is a CAD import — the mount is a fabricated
 * bracket and no drawing of it exists in this repository, so it is drawn as a
 * plausible bracket rather than a fictional exact one.
 */
export function ProbeAssembly({ dimmed, contact, showFrames }: Props) {
  const opacity = dimmed ? 0.45 : 1;
  return (
    <group>
      <AdapterPlate opacity={opacity} />
      <Px6dSensor opacity={opacity} />
      <ProbeMount opacity={opacity} />
      <ConvexProbe opacity={opacity} contact={contact} />
      <ProbeCable opacity={opacity} />
      {showFrames && !dimmed ? <SensorFrame /> : null}
    </group>
  );
}

/** Rigid adapter between the J6 face and the sensor. */
function AdapterPlate({ opacity }: { opacity: number }) {
  return (
    <group>
      <mesh position={[0, 0, STACK.adapter / 2]} rotation={[Math.PI / 2, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.0405, 0.0405, STACK.adapter, 44]} />
        <meshStandardMaterial
          color={ALUMINIUM}
          roughness={0.45}
          metalness={0.12}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
      <BoltCircle z={0.0015} radius={0.031} count={6} opacity={opacity} />
    </group>
  );
}

/**
 * PX6D six-axis force/torque sensor.
 *
 * A 75 mm body, 23 mm thick, bolted through on both faces. The cable leaves
 * the side — that cable is the reason this sensor cannot reach the robot
 * controller, and why the bridge reads it over USB instead.
 */
function Px6dSensor({ opacity }: { opacity: number }) {
  const mid = STACK.sensorFace + STACK.sensor / 2;
  return (
    <group rotation={[0, 0, (SENSOR_YAW_DEG * Math.PI) / 180]}>
      <mesh position={[0, 0, mid]} rotation={[Math.PI / 2, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.0375, 0.0375, STACK.sensor, 48]} />
        <meshStandardMaterial
          color={GRAPHITE}
          roughness={0.4}
          metalness={0.6}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
      {/* Waist groove: the strain-gauge section, and the visual cue that this
          is an instrument rather than a spacer. */}
      <mesh position={[0, 0, mid]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.0345, 0.0345, 0.008, 48]} />
        <meshStandardMaterial
          color="#43463f"
          roughness={0.5}
          metalness={0.12}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
      <BoltCircle z={STACK.sensorFace + 0.0022} radius={0.029} count={6} opacity={opacity} />
      <BoltCircle z={STACK.mountFace - 0.0022} radius={0.029} count={6} opacity={opacity} />
      {/* Connector shell on the side. */}
      <mesh position={[0.0385, 0, mid]} rotation={[0, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.0055, 0.0055, 0.013, 20]} />
        <meshStandardMaterial
          color="#4d504a"
          roughness={0.5}
          metalness={0.5}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
    </group>
  );
}

/**
 * The fabricated bracket, from the supplied solid.
 *
 * Its own datum is the underside of the round base plate — the face that bolts
 * to the PX6D — so it is simply placed at the sensor's top face. The clamp is
 * drawn at 45° in the CAD; rotating by the difference puts its long axis on
 * the probe's image plane, where the assembly actually holds it.
 *
 * The taper is bored down the middle (Ø ~11 mm, measured off the solid) and
 * the probe's cable tail runs down inside it. That bore is why the probe's
 * strain relief does not collide with anything at 90 mm from the flange.
 */
function ProbeMount({ opacity }: { opacity: number }) {
  const geometry = useLoader(STLLoader, MESH_URL('probe_mount_v2')) as BufferGeometry;
  return (
    <mesh
      geometry={geometry}
      position={[0, 0, STACK.mountFace]}
      rotation={[0, 0, ((PROBE_YAW_DEG - MOUNT_CLAMP_CAD_DEG) * Math.PI) / 180]}
      castShadow
      receiveShadow
    >
      <meshStandardMaterial
        color={ALUMINIUM}
        roughness={0.48}
        metalness={0.12}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </mesh>
  );
}

/**
 * GE 4C-RS convex probe, from the manufacturer's solid.
 *
 * The mesh arrives with its origin on the array face and its axes already in
 * the probe convention (§4.1): +z into tissue, +x along the array, +y
 * elevational. So the only thing said here is where that face goes — 234 mm
 * from the flange — and which way the image plane lies.
 *
 * Where it sits along the axis is `STACK.total`. The probe is drawn at the
 * measured insertion, which is 23 mm shallower than the solid's own hard stop
 * — see the note on `STACK`.
 */
function ConvexProbe({ opacity, contact }: { opacity: number; contact: boolean }) {
  const geometry = useLoader(STLLoader, MESH_URL('probe_4c_rs')) as BufferGeometry;
  return (
    <group
      position={[0, 0, STACK.total]}
      rotation={[0, 0, (PROBE_YAW_DEG * Math.PI) / 180]}
    >
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          color={PROBE_SHELL}
          roughness={0.46}
          metalness={0.02}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
      <AcousticFace contact={contact} opacity={opacity} />
    </group>
  );
}

/**
 * The acoustic face, laid on the housing's own curve.
 *
 * A thin shell at the fitted radius, spanning the array footprint. It carries
 * the contact state in the same deep green the force trace and the tool frame
 * use, and it is the only part of the probe that changes colour — because it
 * is the only part that touches anything.
 *
 * Drawn as a cylinder wall about the elevational axis, which is what a convex
 * array is: curved across the image plane, straight along the elevation.
 */
function AcousticFace({ contact, opacity }: { contact: boolean; opacity: number }) {
  // The lit face is the console's structural accent, so it turns blue with
  // everything else on entering contact probing. Its own meaning — "this
  // surface is loaded" — is unchanged; it is the palette underneath that moved.
  const palette = usePalette();
  return (
    <mesh
      // No rotation: three.js builds a cylinder about its own +y, which is the
      // elevational axis here, and starts theta at +z, which is the direction
      // the array faces. Centre it one radius back so the arc grazes the face.
      position={[0, 0, -ARRAY_RADIUS]}
    >
      <cylinderGeometry
        args={[
          ARRAY_RADIUS,
          ARRAY_RADIUS,
          ARRAY_WIDTH,
          48,
          1,
          true,
          -ARRAY_HALF_ANGLE,
          ARRAY_HALF_ANGLE * 2,
        ]}
      />
      <meshStandardMaterial
        color={contact ? palette['--green'] : '#7f837e'}
        roughness={0.3}
        metalness={0.04}
        side={2}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </mesh>
  );
}


/**
 * The probe cable, below the bracket only.
 *
 * The tail is part of the probe solid and ends 90 mm from the flange, inside
 * the bracket's Ø11 mm bore — so the run through the bracket is hidden, which
 * is exactly what you see on the bench. **How it leaves the base is not in the
 * CAD we were given**, so nothing is drawn between the bore and the point
 * where it reappears; a routing invented here would be the only part of this
 * assembly that is not measured.
 *
 * What is drawn is the slack below the sensor. A cable drawn taut tells the
 * operator nothing; one with slack says the wrist can turn without dragging.
 */
function ProbeCable({ opacity }: { opacity: number }) {
  const points = useMemo(() => {
    const curve = new CatmullRomCurve3([
      new Vector3(0.006, 0.026, STACK.mountFace + 0.004),
      new Vector3(0.010, 0.036, STACK.mountFace - 0.006),
      new Vector3(0.008, 0.040, STACK.sensorFace + 0.006),
      new Vector3(-0.002, 0.034, -0.010),
      new Vector3(-0.010, 0.030, -0.036),
    ]);
    return curve.getPoints(40).map((v) => [v.x, v.y, v.z] as [number, number, number]);
  }, []);

  return <Line points={points} color={CABLE} lineWidth={3.2} transparent opacity={opacity} />;
}

/**
 * Sensor frame, drawn on the diagonal it is actually mounted at.
 *
 * Shorter and thinner than the probe frame so the two do not compete: the
 * probe frame is what the controller commands in, and this one is where the
 * numbers on the force panel are measured.
 */
function SensorFrame() {
  const angle = (SENSOR_YAW_DEG * Math.PI) / 180;
  const mid = STACK.sensorFace + STACK.sensor / 2;
  return (
    <group position={[0, 0, mid]} rotation={[0, 0, angle]}>
      <Line
        points={[
          [0, 0, 0],
          [0.05, 0, 0],
        ]}
        color={GRAPHITE}
        lineWidth={1.4}
      />
      <Line
        points={[
          [0, 0, 0],
          [0, 0.05, 0],
        ]}
        color={GRAPHITE}
        lineWidth={1.4}
      />
    </group>
  );
}

function BoltCircle({
  z,
  radius,
  count,
  opacity,
}: {
  z: number;
  radius: number;
  count: number;
  opacity: number;
}) {
  const bolts = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => {
        const a = (i / count) * Math.PI * 2;
        return [Math.cos(a) * radius, Math.sin(a) * radius] as [number, number];
      }),
    [count, radius],
  );
  return (
    <group position={[0, 0, z]}>
      {bolts.map(([x, y], i) => (
        <mesh key={i} position={[x, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.0028, 0.0028, 0.004, 12]} />
          <meshStandardMaterial
            color={GRAPHITE}
            roughness={0.35}
            metalness={0.8}
            transparent={opacity < 1}
            opacity={opacity}
          />
        </mesh>
      ))}
    </group>
  );
}
