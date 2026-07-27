const EPSILON = 1e-9;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function finiteNumber(value, label) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${label} must be finite`);
  }
  return numeric;
}

export function playbackRateFactor(
  timeM,
  slowMotion,
  slowMotionEnabled = true,
) {
  if (!slowMotionEnabled) {
    return 1;
  }
  const start = finiteNumber(slowMotion?.startTimeM, "slowMotion.startTimeM");
  const end = finiteNumber(slowMotion?.endTimeM, "slowMotion.endTimeM");
  const multiplier = finiteNumber(
    slowMotion?.rateMultiplier,
    "slowMotion.rateMultiplier",
  );
  if (!(start < end) || !(multiplier > 0 && multiplier <= 1)) {
    throw new Error("slow-motion configuration is invalid");
  }
  return timeM >= start && timeM < end ? multiplier : 1;
}

export function createPlaybackClock(configuration) {
  const firstTimeM = finiteNumber(
    configuration?.firstTimeM,
    "firstTimeM",
  );
  const finalTimeM = finiteNumber(
    configuration?.finalTimeM,
    "finalTimeM",
  );
  const endHoldSeconds = finiteNumber(
    configuration?.endHoldSeconds,
    "endHoldSeconds",
  );
  const slowMotion = Object.freeze({
    startTimeM: finiteNumber(
      configuration?.slowMotion?.startTimeM,
      "slowMotion.startTimeM",
    ),
    endTimeM: finiteNumber(
      configuration?.slowMotion?.endTimeM,
      "slowMotion.endTimeM",
    ),
    rateMultiplier: finiteNumber(
      configuration?.slowMotion?.rateMultiplier,
      "slowMotion.rateMultiplier",
    ),
  });
  const loop = configuration?.loop === true;

  if (!(firstTimeM < finalTimeM)) {
    throw new Error("playback range must increase");
  }
  if (endHoldSeconds < 0) {
    throw new Error("endHoldSeconds cannot be negative");
  }
  playbackRateFactor(firstTimeM, slowMotion, true);

  let holdElapsedSeconds = 0;
  let lastTimeM = null;

  function resetForExternalTime(timeM) {
    const clamped = clamp(
      finiteNumber(timeM, "timeM"),
      firstTimeM,
      finalTimeM,
    );
    holdElapsedSeconds = 0;
    lastTimeM = clamped;
    return clamped;
  }

  function nextRateBoundary(timeM, slowMotionEnabled) {
    if (!slowMotionEnabled) {
      return finalTimeM;
    }
    if (timeM < slowMotion.startTimeM) {
      return Math.min(slowMotion.startTimeM, finalTimeM);
    }
    if (timeM < slowMotion.endTimeM) {
      return Math.min(slowMotion.endTimeM, finalTimeM);
    }
    return finalTimeM;
  }

  function advance(
    inputTimeM,
    deltaSeconds,
    baseRateMPerSecond,
    slowMotionEnabled = true,
  ) {
    let timeM = clamp(
      finiteNumber(inputTimeM, "timeM"),
      firstTimeM,
      finalTimeM,
    );
    let remainingSeconds = Math.max(
      finiteNumber(deltaSeconds, "deltaSeconds"),
      0,
    );
    const baseRate = Math.max(
      finiteNumber(baseRateMPerSecond, "baseRateMPerSecond"),
      0,
    );

    if (lastTimeM === null || Math.abs(lastTimeM - timeM) > EPSILON) {
      holdElapsedSeconds = 0;
    }
    if (baseRate <= EPSILON || remainingSeconds <= EPSILON) {
      const holding = timeM >= finalTimeM;
      lastTimeM = timeM;
      return Object.freeze({
        timeM,
        effectiveRateMPerSecond: holding
          ? 0
          : baseRate * playbackRateFactor(
            timeM,
            slowMotion,
            slowMotionEnabled,
          ),
        holding,
        looped: false,
      });
    }

    let looped = false;
    let holding = false;
    let guard = 0;
    while (remainingSeconds > EPSILON && guard < 32) {
      guard += 1;
      if (timeM >= finalTimeM) {
        timeM = finalTimeM;
        holding = true;
        const holdRemaining = Math.max(
          endHoldSeconds - holdElapsedSeconds,
          0,
        );
        if (remainingSeconds < holdRemaining - EPSILON) {
          holdElapsedSeconds += remainingSeconds;
          remainingSeconds = 0;
          break;
        }
        remainingSeconds = Math.max(remainingSeconds - holdRemaining, 0);
        holdElapsedSeconds = endHoldSeconds;
        if (!loop) {
          remainingSeconds = 0;
          break;
        }
        timeM = firstTimeM;
        holdElapsedSeconds = 0;
        holding = false;
        looped = true;
        continue;
      }

      const factor = playbackRateFactor(
        timeM,
        slowMotion,
        slowMotionEnabled,
      );
      const boundary = Math.max(
        Math.min(
          nextRateBoundary(timeM, slowMotionEnabled),
          finalTimeM,
        ),
        timeM,
      );
      const sourceDistanceM = boundary - timeM;
      if (sourceDistanceM <= 0) {
        timeM = boundary;
        continue;
      }
      const secondsToBoundary = sourceDistanceM / (baseRate * factor);
      if (remainingSeconds < secondsToBoundary - EPSILON) {
        timeM += remainingSeconds * baseRate * factor;
        remainingSeconds = 0;
      } else {
        timeM = boundary;
        remainingSeconds = Math.max(
          remainingSeconds - secondsToBoundary,
          0,
        );
      }
    }
    if (guard >= 32 && remainingSeconds > EPSILON) {
      throw new Error("playback clock failed to consume the frame interval");
    }

    lastTimeM = clamp(timeM, firstTimeM, finalTimeM);
    holding = holding || lastTimeM >= finalTimeM;
    return Object.freeze({
      timeM: lastTimeM,
      effectiveRateMPerSecond: holding
        ? 0
        : baseRate * playbackRateFactor(
          lastTimeM,
          slowMotion,
          slowMotionEnabled,
        ),
      holding,
      looped,
    });
  }

  return Object.freeze({
    firstTimeM,
    finalTimeM,
    slowMotion,
    advance,
    seek: resetForExternalTime,
    factorAt(timeM, enabled = true) {
      return playbackRateFactor(timeM, slowMotion, enabled);
    },
  });
}
