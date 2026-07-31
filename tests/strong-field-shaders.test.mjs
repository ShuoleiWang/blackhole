import assert from "node:assert/strict";
import test from "node:test";

import { binaryTraceFragmentGLSL } from "../src/binary-shaders.js";
import {
  STRONG_FIELD_UNIFORM_ABI,
  createPnEobOrbitAdapter,
  createStrongFieldSpacetimeProvider,
  evaluateKerrSchild3p1,
} from "../src/strong-field-spacetime.js";
import {
  STRONG_FIELD_DIAGNOSTIC_MODES,
  STRONG_FIELD_MAXIMUM_STEP_M,
  STRONG_FIELD_OUTCOMES,
  STRONG_FIELD_UNIFORM_FLOATS,
  STRONG_FIELD_UNIFORM_LAYOUT,
  STRONG_FIELD_UNIFORM_TAIL_FLOATS,
  strongFieldBinaryShaderBundle,
  strongFieldBinaryTraceFragmentWGSL,
  writeStrongFieldUniformTail,
} from "../src/strong-field-shaders.js";

const orbitAdapter = createPnEobOrbitAdapter({
  dynamicsModel: "4PN/EOB-compatible shader test trajectory",
  coordinateFrame: "asymptotically-inertial-kerr-schild-com",
  source: "deterministic shader contract fixture",
  usesSxsGaugeCentroids: false,
  sample(timeM) {
    return {
      bodies: [
        {
          id: "A",
          massM: 0.5,
          positionM: [-6, 0, 0],
          velocityC: [0, 0, 0.12],
          dimensionlessSpin: [0, 0.08, 0],
        },
        {
          id: "B",
          massM: 0.5,
          positionM: [6, 0, 0],
          velocityC: [0, 0, -0.12],
          dimensionlessSpin: [0, -0.04, 0],
        },
      ],
      remnant: {
        id: "R",
        massM: 0.951609417715,
        positionM: [0, 0, 0],
        velocityC: [0, 0, 0],
        dimensionlessSpin: [0, 0.686461676493, 0],
      },
      mergerBlend: Math.min(1, Math.max(0, (timeM + 100) / 100)),
    };
  },
});
const spacetimeProvider = createStrongFieldSpacetimeProvider({
  orbitAdapter,
});

function frame(overrides = {}) {
  return {
    sceneBinaryState: [12, 0.25, 0, -100],
    sceneBinaryMasses: [0.5, 0.5, 0.951609417715, 0],
    sceneStrongFieldUniforms: spacetimeProvider.frameAt(-100).uniforms,
    ...overrides,
  };
}

function matrixVector(matrix, vector) {
  return matrix.map((row) => (
    row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2]
  ));
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

// Exact Schwarzschild Kerr-Schild 3+1 fields along the radial eigenvector.
function schwarzschildKerrSchildAdm(mass, radius) {
  const twoH = 2 * mass / radius;
  return {
    lapse: 1 / Math.sqrt(1 + twoH),
    radialShift: twoH / (1 + twoH),
    gammaCovariant: [
      [1 + twoH, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    gammaInverse: [
      [1 / (1 + twoH), 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  };
}

test("strong-field bundle declares asymmetric WebGPU production policy", () => {
  assert.equal(strongFieldBinaryShaderBundle.id, "binary-strong-field-v1");
  assert.equal(
    strongFieldBinaryShaderBundle.uniforms.requiredFloatCount,
    STRONG_FIELD_UNIFORM_FLOATS,
  );
  assert.equal(strongFieldBinaryShaderBundle.backendPolicy.production, "webgpu");
  assert.equal(
    strongFieldBinaryShaderBundle.backendPolicy.physicalParityRequired,
    false,
  );
  assert.equal(
    strongFieldBinaryShaderBundle.accumulation.mode,
    "linear-hdr-running-average-v1",
  );
  assert.equal(
    strongFieldBinaryShaderBundle.glsl.trace,
    binaryTraceFragmentGLSL,
  );
  assert.notEqual(
    strongFieldBinaryShaderBundle.wgsl.trace,
    strongFieldBinaryShaderBundle.glsl.trace,
  );
  assert.match(
    strongFieldBinaryShaderBundle.labels.webglFallback,
    /weak-field/i,
  );
});

test("WebGL2 fallback receives only the legacy weak-field ABI", () => {
  class Vector4 {
    fromArray(value) {
      this.value = Array.from(value);
      return this;
    }
  }
  const uniforms = strongFieldBinaryShaderBundle.uniforms.createWebGLExtras({
    Vector4,
  });
  strongFieldBinaryShaderBundle.uniforms.writeWebGLExtras(
    uniforms,
    frame(),
  );
  assert.deepEqual(
    uniforms.uSceneBinaryState.value.value,
    frame().sceneBinaryState,
  );
  assert.deepEqual(
    uniforms.uSceneBinaryMasses.value.value,
    frame().sceneBinaryMasses,
  );
  assert.equal("uSceneStrongFieldUniforms" in uniforms, false);
});

test("uniform ABI is aligned, contiguous, and packs deterministic defaults", () => {
  assert.equal(STRONG_FIELD_UNIFORM_FLOATS % 4, 0);
  assert.equal(STRONG_FIELD_UNIFORM_TAIL_FLOATS, 60);
  assert.equal(STRONG_FIELD_UNIFORM_ABI.floatCount, 44);
  assert.deepEqual(
    Object.values(STRONG_FIELD_UNIFORM_LAYOUT).map(({ offset }) => offset),
    [0, 36, 80, 84, 88, 92],
  );

  const tail = new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS);
  writeStrongFieldUniformTail(tail, frame());
  assert.deepEqual(
    Array.from(tail.slice(0, 44)),
    Array.from(frame().sceneStrongFieldUniforms),
  );
  assert.deepEqual(Array.from(tail.slice(4, 8)), [
    -6,
    0,
    0,
    0.5,
  ]);
  assert.deepEqual(Array.from(tail.slice(8, 12)), [
    0,
    0,
    Math.fround(0.12),
    1,
  ]);
  assert.ok(tail[44] > 0);
  assert.ok(tail[45] > tail[44]);
  assert.ok(tail[48] > 50);
  assert.ok(tail[51] >= 32);
  assert.deepEqual(Array.from(tail.slice(56, 60)), [0, 1, 0, 1]);
});

test("explicit strong-field controls replace defaults without changing ABI", () => {
  const controls = {
    sceneStrongIntegrator: [0.01, 0.3, 2.8, 0.04],
    sceneStrongDomain: [120, 300, 0.02, 96],
    sceneStrongDiagnostics: [5, 220, 0.3, 3],
    strongFieldQuality: {
      accumulationIndex: 7,
      accumulationWeight: 0.125,
      historyEpoch: 4,
      historyReset: false,
    },
  };
  const tail = new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS);
  writeStrongFieldUniformTail(tail, frame(controls));
  const expected = [
    ...frame().sceneStrongFieldUniforms,
    ...controls.sceneStrongIntegrator,
    ...controls.sceneStrongDomain,
    ...controls.sceneStrongDiagnostics,
    7,
    0.125,
    4,
    0,
  ].map(Math.fround);
  assert.deepEqual(Array.from(tail), expected);
});

test("uniform writer fails closed on malformed source state", () => {
  assert.throws(
    () => writeStrongFieldUniformTail(
      new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS),
      frame({ sceneStrongFieldUniforms: new Float32Array(43) }),
    ),
    /44 finite PN\/EOB-provider floats/,
  );
  assert.throws(
    () => writeStrongFieldUniformTail(
      new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS - 1),
      frame(),
    ),
    /needs 60 floats/,
  );
  assert.throws(
    () => writeStrongFieldUniformTail(
      new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS),
      frame({ sceneStrongIntegrator: [0, 0, Number.NaN, 1] }),
    ),
    /sceneStrongIntegrator/,
  );
});

test("WGSL exposes the strong-field provider and complete ray-result contract", () => {
  for (const token of [
    "struct SpacetimeProviderInput",
    "fn sampleSpacetime(",
    "boostedKerrSchildContribution",
    "bodyAPositionMass",
    "bodyAVelocityActive",
    "bodyASpin",
    "attenuationWeight",
    "contractionCoefficient",
    "transformedFactor",
    "observerCameraDirection",
    "spatialMetricDot",
    "asymptoticEscapeDirection",
    "binaryActive",
    "remnantActive",
    "struct ADMFields",
    "fn hamiltonianRhs(",
    "Reduced 3+1 null Hamiltonian",
    "RAY_CAPTURED",
    "RAY_ESCAPED",
    "RAY_UNRESOLVED",
    "escapeDirection",
    "frequencyShift",
    "lookback",
    "nullResidual",
    "minimumHorizonDistance",
    "terminationReason",
    "fn numericalCaptureGuard()",
    "MAX_STRONG_STEPS: i32 = 320",
    "fn accumulationJitter()",
    "radicalInverse(sequenceIndex, 2u)",
    "fn fsMain(",
  ]) {
    assert.ok(
      strongFieldBinaryTraceFragmentWGSL.includes(token),
      `missing WGSL contract token: ${token}`,
    );
  }
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /var<storage|transfer.?map/i,
  );
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /sceneBinaryState|sceneBinaryMasses|orbital phase/i,
  );
  assert.match(strongFieldBinaryTraceFragmentWGSL, /value == value/);
  assert.match(strongFieldBinaryTraceFragmentWGSL, /abs\(value\) < 1\.0e18/);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /stepIndex >= allowedSteps/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /baseBudget \+ criticalBonus/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    new RegExp(
      String.raw`params\.sceneStrongIntegrator\.y,[\s\S]*?`
        + STRONG_FIELD_MAXIMUM_STEP_M.toFixed(1).replace(".", String.raw`\.`),
    ),
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /observerQ \/ max\(conservedEnergy/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /result\.outcome = RAY_UNRESOLVED/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /return mix\(\s*0\.95,\s*0\.25,/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /momentum = -\(observerFields\.spatialMetric \* initialDirection\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /momentum = momentumBeforeKick - acceptedStepSize \* rhs\.momentumRate/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /acceptedStepSize = max\(minimumStep, stepSize \* 0\.25\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /position = position - acceptedStepSize \* driftKinematics\.velocity/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /dot\(position, -rhs\.velocity\) > 0\.0/,
  );
});

test("photographic sky sampling uses a path-independent stable four-tap footprint", () => {
  const qualityStart = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "fn skyQualityPressure()",
  );
  const start = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "fn sampleEnvironment(",
  );
  const end = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "\n}\n\nfn viridis(",
    start,
  );
  assert.ok(qualityStart >= 0 && start > qualityStart && end > start);
  const reconstruction = strongFieldBinaryTraceFragmentWGSL.slice(
    qualityStart,
    end,
  );
  const environment = strongFieldBinaryTraceFragmentWGSL.slice(start, end);
  assert.equal(
    (environment.match(/textureSampleLevel\(/g) || []).length,
    5,
    "one centre sample plus four footprint taps are required",
  );
  assert.match(environment, /sourceFootprint/);
  assert.match(environment, /horizontalFov/);
  assert.match(environment, /footprintPressure/);
  assert.match(environment, /sourceFootprint \* mix\(0\.72, 1\.08, qualityPressure\)/);
  assert.match(environment, /uv \+ vec2<f32>\(radius \* texel\.x, 0\.0\)/);
  assert.match(environment, /uv - vec2<f32>\(0\.0, radius \* texel\.y\)/);
  assert.match(environment, /mix\(centre, filtered, filterWeight\)/);
  assert.match(reconstruction, /params\.renderControls\.w/);
  assert.doesNotMatch(reconstruction, /result\.iterations/);
  assert.doesNotMatch(reconstruction, /result\.minimumHorizonDistance/);
  assert.doesNotMatch(reconstruction, /result\.lookback/);
  assert.doesNotMatch(reconstruction, /result\.terminationReason/);
  assert.match(
    environment,
    /smoothstep\(0\.62, 1\.35, sourceFootprint\)/,
  );
});

test("photographic sky keeps unresolved rays subtle while outcome mode stays vivid", () => {
  const start = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "if (result.outcome == RAY_UNRESOLVED) {",
  );
  const end = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "\n  let shiftRadiance",
    start,
  );
  assert.ok(start >= 0 && end > start);
  const skyUnresolved = strongFieldBinaryTraceFragmentWGSL.slice(start, end);
  assert.match(
    skyUnresolved,
    /vec3<f32>\(0\.050, 0\.036, 0\.024\)/,
  );
  assert.doesNotMatch(
    skyUnresolved,
    /vec3<f32>\(0\.72, 0\.04, 0\.44\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /vec3<f32>\(0\.95, 0\.19, 0\.62\)/,
    "outcome diagnostics must retain a conspicuous failure colour",
  );
});

test("progressive jitter is deterministic across epochs and bounded near the centre", () => {
  const start = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "fn accumulationJitter()",
  );
  const end = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "\n}\n\n@fragment",
    start,
  );
  assert.ok(start >= 0 && end > start);
  const jitter = strongFieldBinaryTraceFragmentWGSL.slice(start, end);
  assert.match(jitter, /let sequenceIndex = sampleIndex;/);
  assert.match(jitter, /let jitterAmplitude = mix\(/);
  assert.match(jitter, /0\.20,\s*0\.58,/);
  assert.doesNotMatch(jitter, /sceneStrongQuality\.z|epoch \*|257u/);
  assert.equal(
    strongFieldBinaryShaderBundle.accumulation.jitter,
    "deterministic-bounded-halton-2-3",
  );
});

test("far-zone step ceiling deliberately rejects the unsafe 4.40 M emergency request", () => {
  assert.equal(STRONG_FIELD_MAXIMUM_STEP_M, 3.5);
  assert.ok(STRONG_FIELD_MAXIMUM_STEP_M < 4.4);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /params\.sceneStrongIntegrator\.y,[\s\S]*?3\.5/,
  );
});

test("diagnostic and outcome enums are stable and non-overlapping", () => {
  assert.deepEqual(STRONG_FIELD_OUTCOMES, {
    unresolved: 0,
    captured: 1,
    escaped: 2,
  });
  assert.deepEqual(STRONG_FIELD_DIAGNOSTIC_MODES, {
    sky: 0,
    outcome: 1,
    lookback: 2,
    frequencyShift: 3,
    hamiltonianResidual: 4,
    integrationCost: 5,
  });
  assert.equal(
    new Set(Object.values(STRONG_FIELD_DIAGNOSTIC_MODES)).size,
    6,
  );
});

test("single-hole Schwarzschild limit has the exact Kerr-Schild ADM values", () => {
  const mass = 1;
  const radius = 10;
  const fields = schwarzschildKerrSchildAdm(mass, radius);
  assert.ok(Math.abs(fields.lapse - (1 / Math.sqrt(1.2))) < 1e-14);
  assert.ok(Math.abs(fields.radialShift - (1 / 6)) < 1e-14);
  assert.ok(
    Math.abs(fields.gammaInverse[0][0] - (5 / 6)) < 1e-14,
  );

  const identity = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const product = fields.gammaCovariant.map((row, rowIndex) => (
    fields.gammaInverse[0].map((_, columnIndex) => (
      row.reduce(
        (sum, value, index) => (
          sum + value * fields.gammaInverse[index][columnIndex]
        ),
        0,
      )
    )).map((value, columnIndex) => value - identity[rowIndex][columnIndex])
  ));
  assert.ok(product.flat().every((value) => Math.abs(value) < 1e-14));
});

test("local momentum construction starts on the 3+1 null cone", () => {
  const fields = schwarzschildKerrSchildAdm(1, 14);
  const direction = [-1, 0.17, 0.08];
  const norm = Math.hypot(...direction);
  const unit = direction.map((value) => value / norm);
  let momentum = matrixVector(fields.gammaCovariant, unit);
  const q0 = Math.sqrt(dot(
    momentum,
    matrixVector(fields.gammaInverse, momentum),
  ));
  momentum = momentum.map((value) => value / q0);
  const q = Math.sqrt(dot(
    momentum,
    matrixVector(fields.gammaInverse, momentum),
  ));
  const shift = [fields.radialShift, 0, 0];
  const energy = fields.lapse * q - dot(shift, momentum);
  const constraint = (
    -((energy + dot(shift, momentum)) ** 2) / (fields.lapse ** 2)
    + q ** 2
  );
  assert.ok(Math.abs(q - 1) < 1e-14);
  assert.ok(Math.abs(constraint) < 1e-14);
});

test("camera FOV is measured in the local ADM orthonormal frame", () => {
  const fields = schwarzschildKerrSchildAdm(1, 8);
  const metricDot = (a, b) => dot(
    a,
    matrixVector(fields.gammaCovariant, b),
  );
  const normalizeMetric = (value) => {
    const inverseNorm = 1 / Math.sqrt(metricDot(value, value));
    return value.map((component) => component * inverseNorm);
  };
  const forward = normalizeMetric([-1, 0.18, 0.04]);
  const rightSeed = [0.02, 0.07, 1];
  const projection = metricDot(forward, rightSeed);
  const right = normalizeMetric(
    rightSeed.map((value, index) => value - projection * forward[index]),
  );
  assert.ok(Math.abs(metricDot(forward, forward) - 1) < 2e-15);
  assert.ok(Math.abs(metricDot(right, right) - 1) < 2e-15);
  assert.ok(Math.abs(metricDot(forward, right)) < 2e-15);

  const angle = 0.42;
  const localRay = normalizeMetric(forward.map(
    (value, index) => value + Math.tan(angle) * right[index],
  ));
  const measured = Math.acos(metricDot(forward, localRay));
  assert.ok(Math.abs(measured - angle) < 2e-15);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /rawRight[\s\S]*spatialMetricDot/,
  );
});

test("finite escape sphere receives the closed-form monopole tail", () => {
  const position = [30, 40, 0];
  const direction = [1, 0, 0];
  const mass = 1;
  const radius = Math.hypot(...position);
  const longitudinal = dot(position, direction);
  const impact = position.map(
    (value, index) => value - longitudinal * direction[index],
  );
  const impactSquared = dot(impact, impact);
  const remainingFraction = 1 - longitudinal / radius;
  const correction = impact.map(
    (value) => -2 * mass * value * remainingFraction / impactSquared,
  );
  assert.ok(Math.abs(correction[0]) < 1e-16);
  assert.ok(Math.abs(correction[1] - (-0.02)) < 1e-16);
  assert.ok(Math.abs(correction[2]) < 1e-16);
  const raw = direction.map((value, index) => value + correction[index]);
  const norm = Math.hypot(...raw);
  const asymptotic = raw.map((value) => value / norm);
  assert.ok(asymptotic[1] < 0);
  assert.ok(Math.abs(Math.hypot(...asymptotic) - 1) < 2e-16);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /remainingFraction[\s\S]*-2\.0 \* asymptoticMass \* impact/,
  );
});

test("Kerr remnant horizon remains sub-extremal at the SXS anchor", () => {
  const mass = 0.951609417715;
  const chi = 0.686461676493;
  const horizonRadius = mass * (1 + Math.sqrt(1 - chi ** 2));
  assert.ok(horizonRadius > mass);
  assert.ok(horizonRadius < 2 * mass);
  assert.ok(Number.isFinite(horizonRadius));
});

test("moving Schwarzschild oracle fixes the Lorentz-covector sign", () => {
  const fields = evaluateKerrSchild3p1({
    massM: 0.5,
    centreM: [-4, 0, 0],
    velocityC: [0, 0, 0.2],
    positionM: [1, 2, 3],
  });
  assert.ok(Math.abs(fields.lapse - 0.9380057452049883) < 2e-14);
  const expectedShift = [
    0.1054368599126387,
    0.0421747439650555,
    0.0392330715550963,
  ];
  expectedShift.forEach((expected, index) => {
    assert.ok(Math.abs(fields.shift[index] - expected) < 2e-14);
  });
  assert.ok(
    Math.abs(fields.covariantMetric[0][0] - (-0.8634488043238141))
      < 2e-14,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /dualSub\(dualConstant\(1\.0\), velocityDotDirection\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /dualSub\(\s*dualScale\(\s*velocityDotDirection/,
  );
});
