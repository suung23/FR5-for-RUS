/**
 * Preload.
 *
 * Intentionally empty of bridges. The renderer needs nothing from the main
 * process, so exposing an IPC channel would only widen the attack surface of a
 * window that displays safety-relevant readings. It exists so that
 * `contextIsolation` has a preload to load, and as the single place to add a
 * bridge later if one is ever genuinely required.
 */
export {};
