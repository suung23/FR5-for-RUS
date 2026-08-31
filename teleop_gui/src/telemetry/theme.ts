import { useSyncExternalStore } from 'react';

/**
 * Console palette, and the one switch that changes it.
 *
 * The console has two palettes. The resting one is the green clinical scheme
 * described in `styles/global.css`. The second is blue, and it means exactly
 * one thing: **the control stack has entered contact probing.**
 *
 * Why a whole-console colour change rather than a badge
 * -----------------------------------------------------
 * Contact probing is not a reading, it is a different machine. The velocity
 * limits drop by fifteen times, the robot closes a force loop on the
 * penetration axis by itself, and the operator's other five axes are commanded
 * to zero. A hand movement that meant one thing a second ago now means
 * something else. That is not a state a field in the right-hand column can
 * carry on its own — an operator watching the arm, not the console, will not
 * be looking at the field when it changes.
 *
 * A full-surface change is seen without being looked at. It is also, unlike a
 * flash or a chime, still there thirty seconds later for someone who walked in
 * mid-procedure.
 *
 * **Colour is still not the only carrier.** The banner says it in words, the
 * event log records it, and the right-hand column keeps its `Velocity limits`
 * field. The rule from `global.css` holds — nothing here is *only* a colour —
 * and the palette is a second, ambient statement of what those already say.
 *
 * How it is wired
 * ---------------
 * `global.css` owns both palettes as custom properties; this module only flips
 * `data-probing` on the root element and republishes what the browser then
 * resolves. So the CSS remains the single definition of every colour, and the
 * canvas-drawn views — the chart, the 3D workspace — read the same values that
 * the DOM ones inherit rather than keeping a second copy that can drift.
 */

/** Tokens the canvas-drawn views need. Everything else reads them via CSS. */
const TOKENS = [
  '--accent',
  '--green',
  '--rule',
  '--panel-pale',
  '--nav-selected',
  '--sidebar',
  '--critical',
  '--ink',
] as const;

type Token = (typeof TOKENS)[number];
export type Palette = Record<Token, string>;

/**
 * Values to fall back on if a token resolves empty.
 *
 * Only reachable before the stylesheet has been applied. They are the green
 * resting palette, so a console that somehow renders without CSS is the
 * *resting* colour rather than the probing one — the failure mode has to be
 * "looks normal while probing", never "looks like probing while approaching".
 */
const FALLBACK: Palette = {
  '--accent': '#4b8b5a',
  '--green': '#17683a',
  '--rule': '#b9d0be',
  '--panel-pale': '#f1f8f2',
  '--nav-selected': '#d9ecdc',
  '--sidebar': '#f6fbf7',
  '--critical': '#8b1e1e',
  '--ink': '#000000',
};

function read(): Palette {
  if (typeof document === 'undefined') return FALLBACK;
  const computed = getComputedStyle(document.documentElement);
  const out = {} as Palette;
  for (const token of TOKENS) {
    const value = computed.getPropertyValue(token).trim();
    out[token] = value.length > 0 ? value : FALLBACK[token];
  }
  return out;
}

let palette: Palette = read();
const listeners = new Set<() => void>();

/**
 * Put the console into — or out of — the contact-probing palette.
 *
 * Idempotent: called on every telemetry frame, does work only on a change.
 */
export function setProbingTheme(on: boolean): void {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  const current = root.getAttribute('data-probing') === 'on';
  if (current === on) return;

  if (on) root.setAttribute('data-probing', 'on');
  else root.removeAttribute('data-probing');

  // Re-read rather than compute: the attribute has changed, so the browser has
  // already resolved the other palette, and asking it is what keeps this file
  // from holding a second copy of the colours.
  palette = read();
  for (const listener of listeners) listener();
}

/**
 * Re-resolve the palette from the document.
 *
 * The module-level read happens when this file is first evaluated, which is
 * before React has mounted anything. Calling this once from the app's mount
 * effect means the published values are the ones a fully built document
 * resolves, whatever order the bundler happened to evaluate modules in.
 */
export function refreshPalette(): void {
  const next = read();
  const changed = (Object.keys(next) as (keyof Palette)[]).some(
    (token) => next[token] !== palette[token],
  );
  if (!changed) return;
  palette = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** The palette in force, for views that draw rather than style. */
export function usePalette(): Palette {
  return useSyncExternalStore(subscribe, () => palette, () => palette);
}
