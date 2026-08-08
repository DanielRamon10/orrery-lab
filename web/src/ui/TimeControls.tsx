/**
 * Play/pause, speed, and a scrub bar across three centuries.
 *
 * The slider's range is deliberately 1800–2100. The element table's published
 * validity is 1800–2050, and it degrades gracefully rather than sharply past that,
 * so a little overshoot is honest while a slider reaching year 10 000 would not be.
 */

import { dateFromJulianDate, julianDateFromDate } from "../lib/ephemeris";
import { SPEED_PRESETS, useSimulation, useSimulationTime } from "../state/simulation";

const SCRUB_START_JD = julianDateFromDate(new Date("1800-01-01T00:00:00Z"));
const SCRUB_END_JD = julianDateFromDate(new Date("2100-01-01T00:00:00Z"));

const DATE_FORMAT = new Intl.DateTimeFormat("en-GB", {
  year: "numeric",
  month: "short",
  day: "2-digit",
  timeZone: "UTC",
});

export function TimeControls() {
  const { seek, seekToNow, setPlaying, setDaysPerSecond } = useSimulation();
  const { jd, playing, daysPerSecond } = useSimulationTime();

  const moment = dateFromJulianDate(jd);
  const withinTable = jd >= SCRUB_START_JD && jd <= julianDateFromDate(new Date("2050-01-01Z"));

  return (
    <div className="panel panel--time">
      <div className="time-row">
        <button
          type="button"
          className="control control--primary"
          onClick={() => setPlaying(!playing)}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "❚❚" : "▶"}
        </button>

        <div className="time-readout">
          <span className="time-readout__date">{DATE_FORMAT.format(moment)}</span>
          <span className="time-readout__jd">
            JD {jd.toFixed(3)}
            {!withinTable && <em className="time-readout__warn"> · outside table range</em>}
          </span>
        </div>

        <button type="button" className="control" onClick={seekToNow}>
          Now
        </button>
      </div>

      <input
        type="range"
        className="scrub"
        min={SCRUB_START_JD}
        max={SCRUB_END_JD}
        step={1}
        value={jd}
        onChange={(event) => seek(Number(event.target.value))}
        aria-label="Scrub through time"
      />

      <div className="speed-row">
        <button
          type="button"
          className="control control--compact"
          onClick={() => setDaysPerSecond(-Math.abs(daysPerSecond))}
          aria-label="Run time backwards"
          aria-pressed={daysPerSecond < 0}
        >
          ◀◀
        </button>

        {SPEED_PRESETS.map((preset) => {
          const active = Math.abs(daysPerSecond) === preset.daysPerSecond;
          return (
            <button
              key={preset.label}
              type="button"
              className={`control control--compact${active ? " control--active" : ""}`}
              onClick={() =>
                setDaysPerSecond(Math.sign(daysPerSecond || 1) * preset.daysPerSecond)
              }
              aria-pressed={active}
            >
              {preset.label}
            </button>
          );
        })}

        <button
          type="button"
          className="control control--compact"
          onClick={() => setDaysPerSecond(Math.abs(daysPerSecond))}
          aria-label="Run time forwards"
          aria-pressed={daysPerSecond > 0}
        >
          ▶▶
        </button>
      </div>
    </div>
  );
}
