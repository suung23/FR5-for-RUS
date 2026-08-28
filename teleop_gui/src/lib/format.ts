/**
 * Number formatting for readings that update several times a second.
 *
 * The rule this file exists for: **a value that is changing must not change
 * width.** A cell that gains and loses a sign as the reading crosses zero
 * moves every glyph in its column, and a table of six joints doing that a few
 * times a second reads as the whole panel shaking. The operator sees motion
 * and looks for a fault; there is none, it is the typography.
 */

/** Unicode minus. Same advance width as `+` in the interface face. */
const MINUS = '−';

/**
 * A signed fixed-point reading whose width never changes.
 *
 * The sign is always present, `+0.0` and `−0.0` included. At rest the sign can
 * still flip between them, but the column no longer moves — which is the part
 * that was legible as shaking.
 *
 * @param value the reading, or undefined/null when there is nothing to show
 * @param places decimal places
 */
export function signed(value: number | null | undefined, places: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const rounded = Number(value.toFixed(places));
  // Take the sign from the value, not from the rounded result: a reading of
  // −0.04 at one decimal is −0.0, and printing it as +0.0 would be wrong about
  // the direction it is actually on.
  const negative = rounded < 0 || (rounded === 0 && (value < 0 || Object.is(value, -0)));
  return (negative ? MINUS : '+') + Math.abs(rounded).toFixed(places);
}

/** An unsigned reading, or an em dash when there is nothing to show. */
export function fixed(value: number | null | undefined, places: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toFixed(places);
}
