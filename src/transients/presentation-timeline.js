// Presentation timeline for transients that span many decades of physical
// time (milliseconds of a detonation to months of a light curve).
//
// The timeline is a chain of segments, each covering a physical-time interval
// with a wall-clock duration at nominal playback speed. Inside a "linear"
// segment physical time advances uniformly; inside a "log" segment log10(t)
// does (t measured from the event, t > 0). The same map drives playback and
// the scrubber, so both have usable resolution at every scale. It changes only
// the presentation rate, never the physical state at a given time.

function validateSegments(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    throw new Error("Presentation timeline needs at least one segment");
  }
  let previousEnd = null;
  return segments.map((segment, index) => {
    const start = Number(segment.start);
    const end = Number(segment.end);
    const wallSeconds = Number(segment.wallSeconds);
    const kind = segment.kind;
    if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) {
      throw new RangeError(`Timeline segment ${index} needs start < end`);
    }
    if (!(wallSeconds > 0)) {
      throw new RangeError(`Timeline segment ${index} needs a positive wall duration`);
    }
    if (kind !== "linear" && kind !== "log") {
      throw new RangeError(`Timeline segment ${index} kind must be linear or log`);
    }
    if (kind === "log" && !(start > 0)) {
      throw new RangeError(`Logarithmic timeline segment ${index} must start after t = 0`);
    }
    if (previousEnd !== null && Math.abs(start - previousEnd) > 1e-12 * Math.max(1, Math.abs(start))) {
      throw new RangeError(`Timeline segment ${index} must start where the previous one ends`);
    }
    previousEnd = end;
    return Object.freeze({
      kind,
      start,
      end,
      wallSeconds,
      id: segment.id ?? `segment-${index}`,
    });
  });
}

export function createPresentationTimeline({
  segments: segmentInput,
  endHoldSeconds = 2,
  loop = true,
} = {}) {
  const segments = validateSegments(segmentInput);
  const totalWall = segments.reduce((sum, segment) => sum + segment.wallSeconds, 0);
  const wallStarts = [];
  let accumulated = 0;
  for (const segment of segments) {
    wallStarts.push(accumulated);
    accumulated += segment.wallSeconds;
  }
  const firstTime = segments[0].start;
  const finalTime = segments[segments.length - 1].end;

  function segmentIndexForTime(time) {
    if (time <= firstTime) return 0;
    for (let index = 0; index < segments.length; index += 1) {
      if (time < segments[index].end) return index;
    }
    return segments.length - 1;
  }

  function fractionInSegment(segment, time) {
    const clamped = Math.min(Math.max(time, segment.start), segment.end);
    if (segment.kind === "linear") {
      return (clamped - segment.start) / (segment.end - segment.start);
    }
    return Math.log(clamped / segment.start) / Math.log(segment.end / segment.start);
  }

  function timeInSegment(segment, fraction) {
    const f = Math.min(Math.max(fraction, 0), 1);
    if (segment.kind === "linear") {
      return segment.start + f * (segment.end - segment.start);
    }
    return segment.start * (segment.end / segment.start) ** f;
  }

  function wallAtTime(time) {
    const index = segmentIndexForTime(time);
    return wallStarts[index] + segments[index].wallSeconds * fractionInSegment(segments[index], time);
  }

  function timeAtWall(wall) {
    const w = Math.min(Math.max(wall, 0), totalWall);
    let index = segments.length - 1;
    for (let candidate = 0; candidate < segments.length; candidate += 1) {
      if (w < wallStarts[candidate] + segments[candidate].wallSeconds) {
        index = candidate;
        break;
      }
    }
    const segment = segments[index];
    return timeInSegment(segment, (w - wallStarts[index]) / segment.wallSeconds);
  }

  // Physical seconds per wall second at nominal speed.
  function rateAt(time) {
    const segment = segments[segmentIndexForTime(time)];
    if (segment.kind === "linear") {
      return (segment.end - segment.start) / segment.wallSeconds;
    }
    const clamped = Math.min(Math.max(time, segment.start), segment.end);
    return clamped * Math.log(segment.end / segment.start) / segment.wallSeconds;
  }

  return Object.freeze({
    segments,
    firstTime,
    finalTime,
    totalWallSeconds: totalWall,
    endHoldSeconds,
    loop,
    progressAtTime(time) {
      return wallAtTime(time) / totalWall;
    },
    timeAtProgress(progress) {
      return timeAtWall(progress * totalWall);
    },
    segmentAt(time) {
      return segments[segmentIndexForTime(time)];
    },
    rateAt,
    seek(time) {
      return Math.min(Math.max(Number(time), firstTime), finalTime);
    },
    /**
     * Advance by a wall-clock interval at a playback speed multiplier.
     * `holdElapsed` carries the end-of-track hold between calls.
     */
    advance(time, wallDelta, speed = 1, holdElapsed = 0) {
      if (!(wallDelta >= 0) || !(speed >= 0)) {
        throw new RangeError("Timeline advance needs non-negative wall delta and speed");
      }
      if (time >= finalTime) {
        const held = holdElapsed + wallDelta;
        if (loop && held >= endHoldSeconds) {
          return { time: firstTime, holding: false, holdElapsed: 0, rate: 0, wrapped: true };
        }
        return { time: finalTime, holding: true, holdElapsed: held, rate: 0, wrapped: false };
      }
      const wall = wallAtTime(time) + wallDelta * speed;
      if (wall >= totalWall) {
        return { time: finalTime, holding: true, holdElapsed: 0, rate: 0, wrapped: false };
      }
      const nextTime = timeAtWall(wall);
      return {
        time: nextTime,
        holding: false,
        holdElapsed: 0,
        rate: rateAt(nextTime) * speed,
        wrapped: false,
      };
    },
  });
}
