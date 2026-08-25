/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ROBOT_HOST?: string;
  readonly VITE_ROBOT_TELEMETRY_TRANSPORT?: string;
  readonly VITE_ROBOT_TELEMETRY_URL?: string;
  readonly VITE_WRENCH_URL?: string;
  readonly VITE_HTTP_POLL_MS?: string;
  readonly VITE_ROS_JOINT_TOPIC?: string;
  readonly VITE_ROS_WRENCH_TOPIC?: string;
  readonly VITE_CONTACT_ENTER_N?: string;
  readonly VITE_CONTACT_RELEASE_N?: string;
  readonly VITE_WARN_FORCE_N?: string;
  readonly VITE_MAX_FORCE_N?: string;
  readonly VITE_NORMAL_FORCE_SIGN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
