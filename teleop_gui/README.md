# Robotic Ultrasound Teleoperation Monitor

A desktop telemetry monitor for the FR5 right-arm ultrasound probing cell.
It watches robot and force-sensor telemetry, draws the arm from the joint
angles, shows which operating stage the system is in, and gives the operator
short guidance tied to what it is actually reading.

> **Research use only.** This is not a medical device. It is not cleared or
> approved for diagnosis or treatment, and it does not support autonomous
> clinical operation. The application is **receive-only**: it has no command
> path to the robot, and adding one is out of scope — motion belongs to the
> ROS 2 stack that owns the velocity clamps and watchdogs.

![Monitoring view](.screens/monitoring.png)

## Running it

```bash
npm install
npm run console          # Electron window on the workstation's display
npm run build            # typecheck + production build
```

From an SSH session `npm run console` still opens the window on the machine's
own monitor — [`scripts/run-on-console.sh`](scripts/run-on-console.sh) supplies
`DISPLAY` and the X credentials, and refuses with a clear message rather than
starting blind if the seat is not there.

To review the interface in a browser instead: `npx vite --port 5173`.

## Connecting to the robot

The console follows a real session. It does not generate motion, and with no
source connected it shows **NO TELEMETRY** and a still arm.

Start the bridge on the workstation running the Phase 0 stack:

```bash
ros2 run fr5_control telemetry_bridge
```

It subscribes to the stack's own topics and republishes them in this contract:

| Topic | Carries |
|---|---|
| `/fr5_right/joint_states` | joint positions, velocities |
| `/fr5_right/ee_wrt_base` | TCP pose |
| `/fr5_right/wrench` | force/torque, when the controller has a sensor |

The bridge is **subscribe-only**. It publishes nothing and discards anything
arriving on the socket; the single path that moves the robot stays with the
control stack that owns the velocity clamps and watchdogs.

### Force

Our PX6D is the USB variant, so `robot_state_pkg.ft_sensor_data` is empty and
`/fr5_right/wrench` carries nothing real. Point the bridge at the sensor
directly:

```bash
sg dialout -c "ros2 run fr5_control telemetry_bridge --ros-args \
  -p bridge.px6d_port:=/dev/ttyACM0"
```

Each wrench frame reports its `source`, and the console names it — the header
reads `PX6D · USB DIRECT` or `VIA CONTROLLER` rather than claiming PX6D
whatever is feeding it. When the sensor is read directly the plate also shows
the measured sample rate and the CRC count since the stream synchronised, so
"the numbers look wrong" and "the cable is bad" can be told apart without
leaving the console.

### Other transports

`websocket` (the bridge, default), `http`, `rosbridge`, and `simulation`. See
[`.env.example`](.env.example). `rosbridge` needs `rosbridge_suite`, which is
not installed on this workstation. `simulation` generates telemetry for
reviewing the interface without hardware and must be asked for explicitly —
every panel labels it while it is on.

## Visual system

An operational console in the FAIRINO FR-series idiom: black canvas, white
information plates, square corners, 1px rules. Three colours carry everything —
black `#000000`, white `#ffffff`, deep navy `#10233f` — with `#d6d9de` for rules
and `#6b7280` for de-emphasised text.

Navy is reserved: the selected navigation item, the active tab underline,
section headers, the measured value on the force scale, and the joints and
flange path in the workspace. Nothing else is coloured.

**No state is signalled by colour alone.** Severity is structural:

| Level | How it reads |
|---|---|
| normal | plain row, hairline rule |
| warning | 2px black left edge, `WARN` tag, heavier rule |
| alarm | inverted block — white on black — plus an `ALARM` tag |

That survives a monochrome pendant, a photocopy and colour-blind vision. The
trade-off is real and worth stating: an inverted block is less instantly
arresting at a distance than a red lamp, so the console leans on the event log
and the guidance column to carry urgency in words.

## Layout

```
SYSTEM │ ROBOT: CONNECTED │ SENSOR: CONNECTED │ MODE: TELEOPERATION │ SOURCE
───────┬──────────────────────────────────────────────┬────────────────────
 VIEW  │ Robot workspace                              │ Operating stage
 Monit.│ FR5 + TCP frame + 10 s flange path           │ Normal force
 Teleo.├──────────────────────────────────────────────┤ TCP pose
 Contact│ force trend │ joint travel │ mode timeline  │ Joint state
 Safety │            │ safety limits                  │ Force / torque
───────┴──────────────────────────────────────────────┴ Operator guidance
 EVENT LOG · timestamps · telemetry age · research-use-only
```

The four navigation entries are operational views, not decoration — each
changes the lower workspace panel:

| View | Lower panel |
|---|---|
| Monitoring | rolling force trend, normal or per-axis components |
| Teleoperation | joint travel within each axis' own limits |
| Contact | stage-transition timeline reconstructed from the buffer |
| Safety | configured limits and standing caveats |

The selected view is held in the URL hash, so a reload returns to the panel the
operator was on.

## Architecture

```
transports/{websocket,http,rosbridge,simulation}
        │  parse + validate
        ▼
RobotTelemetryAdapter          ← the only seam; receive-only by construction
        │  RobotTelemetry | WrenchSample | LinkStatus
        ▼
zustand store  ─→  ContactDetector (port of contact_state.py)
        │
        ▼
React console (workspace · trend · timeline · numeric column · event log)
```

Everything above the adapter works against the shared contract and cannot tell
which transport is underneath. Swapping the simulator for a live bridge changes
where numbers come from, never what they mean.

| Path | Role |
|---|---|
| `src/telemetry/types.ts` | the shared contract |
| `src/telemetry/adapter.ts` | transport selection, link status, staleness watchdog |
| `src/telemetry/contactState.ts` | approach/contact classification |
| `src/telemetry/fr5Model.ts` | FR5 URDF chain, forward kinematics, joint limits |
| `src/telemetry/guidance.ts` | operator guidance rules |
| `src/store/telemetryStore.ts` | live state, chart decimation, peak hold, flange path |
| `src/store/events.ts` | transition-only event log |

## Thresholds

Defaults mirror `fr5_control/config/probe.yaml` and can be overridden through
the environment. Keep them in step with that file — it is the single source of
truth for the control stack, and this display agreeing with it matters more
than either being independently adjustable.

| Value | Default | From |
|---|---|---|
| contact enter | 7.0 N | `safety.max_normal_force_n` |
| contact release | 0.2 N | `watchdog.retreat_until_force_n` |
| warning | 6.0 N | `safety.warn_normal_force_n` |
| `F_n = sign × F_z` | −1.0 | `ft_sensor.normal_force_sign` |

`ContactDetector` is a direct port of `fr5_control/contact_state.py`, including
the hysteresis, the confirmation windows (5 ms to enter, 50 ms to release) and
the latch. If one changes, change both.

## What the display cannot tell you

These are stated on screen as well, because they change how the numbers should
be read:

- **Probe weight is not compensated.** `payload.mass_kg` is unidentified, so the
  normal force carries a pose-dependent gravity term of roughly 1.5 N. Every
  force threshold inherits that error until `ForceSensorAutoComputeLoad()` has
  been run.
- **The sensor's axis assignment is provisional.** The PX6D manual's §5.3 and
  §5.4 disagree, and `normal_force_sign` has not been verified against hardware.
- **The tool transform is unmeasured.** `tool.j6_to_probe_*` is still `.nan`, so
  the 3D view marks the J6 flange, not the probe contact point.
- **The robot model is Fairino's own.** `public/models/fr5/` holds the seven
  STL parts from `fairino_description`, placed on the URDF chain from
  `fairino5_v6.urdf`. Cross-checked against the robot: at the home pose our
  forward kinematics reproduces the TCP the controller itself reports to
  0.1 mm. The vendor's "STEP Models" download is not usable here — it contains
  SolidWorks native parts, which no open tool reads.

![Safety view](.screens/safety.png)
