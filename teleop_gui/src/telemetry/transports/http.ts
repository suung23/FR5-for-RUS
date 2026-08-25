import type { Transport, TransportSink } from '../types';
import { parseTelemetry, parseWrench } from './parse';

/**
 * Polling transport for endpoints that expose telemetry as a plain GET.
 *
 * Polling is scheduled after each response rather than on a fixed interval, so
 * a slow endpoint cannot accumulate a queue of overlapping requests that would
 * make the display lag further and further behind reality.
 */
export class HttpPollTransport implements Transport {
  readonly kind = 'http' as const;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private controller: AbortController | null = null;
  private stopped = false;
  private attempts = 0;

  constructor(
    readonly endpoint: string,
    private readonly periodMs: number,
    private readonly wrenchUrl?: string,
  ) {}

  start(sink: TransportSink): void {
    this.stopped = false;
    sink.onStatus({ phase: 'connecting', attempts: 0 });
    void this.tick(sink);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.controller?.abort();
    this.controller = null;
  }

  private async tick(sink: TransportSink): Promise<void> {
    if (this.stopped) return;
    this.controller = new AbortController();
    const deadline = setTimeout(() => this.controller?.abort(), this.periodMs * 5);

    try {
      const res = await fetch(this.endpoint, { signal: this.controller.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const now = Date.now();
      const frame = parseTelemetry(await res.json(), now);
      if (frame) {
        sink.onTelemetry({ ...frame, connected: true });
        sink.onStatus({ phase: 'connected', attempts: 0, error: undefined, lastFrameAt: now });
        this.attempts = 0;
      }

      if (this.wrenchUrl) {
        const wres = await fetch(this.wrenchUrl, { signal: this.controller.signal });
        if (wres.ok) {
          const sample = parseWrench(await wres.json(), Date.now());
          if (sample) sink.onWrench(sample);
        }
      }
    } catch (err) {
      if (!this.stopped) {
        this.attempts += 1;
        sink.onTelemetry({ timestamp: Date.now(), connected: false });
        sink.onStatus({
          phase: 'error',
          attempts: this.attempts,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    } finally {
      clearTimeout(deadline);
      this.controller = null;
    }

    if (this.stopped) return;
    // Back off while failing so a dead endpoint is not hammered at 10 Hz.
    const wait = this.attempts > 0 ? Math.min(this.periodMs * 2 ** this.attempts, 5_000)
      : this.periodMs;
    this.timer = setTimeout(() => void this.tick(sink), wait);
  }
}
