import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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
  STRONG_FIELD_ACCRETION_UNIFORM_FLOATS,
  STRONG_FIELD_ACCRETION_UNIFORM_LAYOUT,
  STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS,
  STRONG_FIELD_MAXIMUM_STEP_M,
  STRONG_FIELD_OUTCOMES,
  STRONG_FIELD_UNIFORM_FLOATS,
  STRONG_FIELD_UNIFORM_LAYOUT,
  STRONG_FIELD_UNIFORM_TAIL_FLOATS,
  strongFieldBinaryDualDiskShaderBundle,
  strongFieldBinaryDualDiskTraceFragmentWGSL,
  strongFieldBinaryShaderBundle,
  strongFieldBinaryTraceFragmentWGSL,
  writeStrongFieldAccretionUniformTail,
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

const dualDiskUniforms = Object.freeze([
  1, 0.8, 1, 6,
  0, 1, 0, 3,
  8.5, 6.3e-5, 1, 1,
  0, 1, 0, 3,
  8.5, 6.3e-5, 1, 1,
]);

function dualDiskFrame(overrides = {}) {
  return frame({
    sceneStrongAccretionUniforms: dualDiskUniforms,
    ...overrides,
  });
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
  assert.deepEqual(
    strongFieldBinaryShaderBundle.wgsl.traceSpecializations.map(
      ({ id, constants }) => [id, constants.SPACETIME_PHASE_MODE],
    ),
    [["binary", 0], ["transition", 2], ["remnant", 1]],
  );
  const selectPhase = strongFieldBinaryShaderBundle.wgsl
    .selectTraceSpecialization;
  assert.equal(selectPhase({ sceneStrongFieldUniforms: [0, 0] }), "binary");
  assert.equal(
    selectPhase({ sceneStrongFieldUniforms: [0, 0.5] }),
    "transition",
  );
  assert.equal(selectPhase({ sceneStrongFieldUniforms: [0, 1] }), "remnant");
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

test("dual-disk bundle keeps phase specializations and weak-field fallback explicit", () => {
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.id,
    "binary-dual-disk-strong-field-v1",
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.uniforms.requiredFloatCount,
    STRONG_FIELD_ACCRETION_UNIFORM_FLOATS,
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.wgsl.trace,
    strongFieldBinaryDualDiskTraceFragmentWGSL,
  );
  assert.deepEqual(
    strongFieldBinaryDualDiskShaderBundle.wgsl.traceSpecializations.map(
      ({ id, constants }) => [id, constants.SPACETIME_PHASE_MODE],
    ),
    [["binary", 0], ["transition", 2], ["remnant", 1]],
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.glsl.trace,
    binaryTraceFragmentGLSL,
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.backendPolicy.physicalParityRequired,
    false,
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.backendPolicy.matterBackreaction,
    false,
  );
  assert.match(
    strongFieldBinaryDualDiskShaderBundle.backendPolicy.scientificStatus,
    /analytic.*not GRMHD or NR/i,
  );
  assert.match(
    strongFieldBinaryDualDiskShaderBundle.backendPolicy.scientificStatus,
    /bounded phenomenological emissivity texture/i,
  );
  assert.match(
    strongFieldBinaryDualDiskShaderBundle.labels.webglFallback,
    /vacuum.*without disk parity/i,
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

test("dual-disk ABI appends twenty floats without changing the vacuum ABI", () => {
  assert.equal(STRONG_FIELD_UNIFORM_FLOATS, 96);
  assert.equal(STRONG_FIELD_UNIFORM_TAIL_FLOATS, 60);
  assert.equal(STRONG_FIELD_ACCRETION_UNIFORM_FLOATS, 116);
  assert.equal(STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS, 80);
  assert.deepEqual(
    Object.values(STRONG_FIELD_ACCRETION_UNIFORM_LAYOUT)
      .map(({ offset }) => offset),
    [0, 36, 80, 84, 88, 92, 96],
  );

  const vacuumTail = new Float32Array(STRONG_FIELD_UNIFORM_TAIL_FLOATS);
  const accretionTail = new Float32Array(
    STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS,
  );
  writeStrongFieldUniformTail(vacuumTail, frame());
  writeStrongFieldAccretionUniformTail(accretionTail, dualDiskFrame());
  assert.deepEqual(
    Array.from(accretionTail.slice(0, STRONG_FIELD_UNIFORM_TAIL_FLOATS)),
    Array.from(vacuumTail),
  );
  assert.deepEqual(
    Array.from(accretionTail.slice(STRONG_FIELD_UNIFORM_TAIL_FLOATS)),
    dualDiskUniforms.map(Math.fround),
  );
});

test("dual-disk uniform writer rejects malformed or non-physical transfer state", () => {
  const write = (values, length = STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS) => (
    writeStrongFieldAccretionUniformTail(
      new Float32Array(length),
      dualDiskFrame({ sceneStrongAccretionUniforms: values }),
    )
  );
  assert.throws(
    () => write(dualDiskUniforms, STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS - 1),
    /needs 80 floats/,
  );
  assert.throws(() => write(dualDiskUniforms.slice(0, 19)), /20 finite/);
  assert.throws(
    () => write(dualDiskUniforms.with(0, 0.5)),
    /active flag/,
  );
  for (const values of [
    dualDiskUniforms.with(1, 0),
    dualDiskUniforms.with(1, 1.01),
    dualDiskUniforms.with(2, 0),
    dualDiskUniforms.with(2, 17),
    dualDiskUniforms.with(3, -1),
    dualDiskUniforms.with(3, 101),
  ]) {
    assert.throws(() => write(values), /control parameters/);
  }
  for (const values of [
    dualDiskUniforms.with(4, 0.2),
    dualDiskUniforms.with(7, 0),
    dualDiskUniforms.with(8, 2.9),
    dualDiskUniforms.with(9, 0),
    dualDiskUniforms.with(10, 1.1),
    dualDiskUniforms.with(11, 0),
    dualDiskUniforms.with(16, 1.0e5 + 1),
    dualDiskUniforms.with(17, 1.0e3 + 1),
  ]) {
    assert.throws(() => write(values), /disk parameters/);
  }
  const inactive = [
    0, 0.8, 1, 6,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
  ];
  assert.doesNotThrow(() => write(inactive));

  const transitionProvider = spacetimeProvider.frameAt(-50).uniforms;
  assert.ok(transitionProvider[1] > 0 && transitionProvider[1] < 1);
  assert.throws(
    () => writeStrongFieldAccretionUniformTail(
      new Float32Array(STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS),
      dualDiskFrame({ sceneStrongFieldUniforms: transitionProvider }),
    ),
    /must be dark during merger transition/,
  );
  const darkTransitionDisks = dualDiskUniforms
    .with(10, 0)
    .with(18, 0);
  assert.doesNotThrow(() => writeStrongFieldAccretionUniformTail(
    new Float32Array(STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS),
    dualDiskFrame({
      sceneStrongFieldUniforms: transitionProvider,
      sceneStrongAccretionUniforms: darkTransitionDisks,
    }),
  ));
});

test("vacuum generated WGSL remains byte-for-byte unchanged", () => {
  assert.equal(
    createHash("sha256").update(strongFieldBinaryTraceFragmentWGSL).digest("hex"),
    "de367f0ba7f2f2bc71750d983067ac813c35eb0818c0b1dfdb0cf2ef7ab28849",
  );
  assert.equal(strongFieldBinaryTraceFragmentWGSL.length, 49097);
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /sceneDiskControl|DiskIntersection|diskRadiance|accumulateDualDiskEmission/,
  );
});

test("dual-disk WGSL sorts segment crossings and composes finite optical depth", () => {
  const shader = strongFieldBinaryDualDiskTraceFragmentWGSL;
  for (const token of [
    "sceneDiskControl: vec4<f32>",
    "struct DiskIntersection",
    "fn segmentDiskIntersection(",
    "let fraction = sideStart / denominator",
    "let firstIsA = hitA.fraction <= hitB.fraction",
    "fn applyDiskIntersection(",
    "result.diskRadiance = result.diskRadiance",
    "result.diskTransmittance = result.diskTransmittance * (1.0 - sample.opacity)",
    "let lineOfSightTau = min(tauFace / muEmitter, 30.0)",
    "let opaqueSurfaceFraction = clamp(",
    "activeWeight * edgeCoverage * opaqueSurfaceFraction",
    "fields.horizonDistance <= capturePadding",
    "result.diskTransferFailure = 1.0",
  ]) {
    assert.ok(shader.includes(token), `missing dual-disk token: ${token}`);
  }
  const firstBranch = shader.slice(
    shader.indexOf("if (firstIsA)"),
    shader.indexOf("return result;", shader.indexOf("if (firstIsA)")),
  );
  assert.ok(firstBranch.indexOf("result, hitA") < firstBranch.indexOf("result, hitB"));
  assert.match(
    shader,
    /let previousPosition = position;[\s\S]*position = position - acceptedStepSize[\s\S]*accumulateDualDiskEmission\(/,
  );
});

test("dual-disk transfer uses local emitter energy, g4, and mass-scaled T_eff4", () => {
  const shader = strongFieldBinaryDualDiskTraceFragmentWGSL;
  for (const pattern of [
    /let emitterFrequency = \([\s\S]*conservedEnergy \* emitterTime - dot\(momentum, emitterSpatial\)/,
    /let rawFrequencyShift = observerFrequency \/ emitterFrequency/,
    /let chromaticFrequencyShift = clamp\(rawFrequencyShift, 0\.02, 8\.0\)/,
    /let g2 = rawFrequencyShift \* rawFrequencyShift/,
    /let bolometricTransfer = g2 \* g2/,
    /eddingtonRatio \* 1\.0e8 \/ bodyMassSolar[\s\S]*structuredFluxShape \* thermalFluxScale/,
    /visibleBlackbodyLinearSrgbPerBolometric\([\s\S]*emittedTemperature \* chromaticFrequencyShift[\s\S]*\)/,
    /if \(!finiteScalar\(rawFrequencyShift\) \|\| rawFrequencyShift <= 0\.0\)/,
  ]) {
    assert.match(shader, pattern);
  }
  assert.match(
    shader,
    /bolometric surface flux retains T_eff\^4 proportional to[\s\S]*\(Mdot \/ M\^2\)/,
  );
  assert.doesNotMatch(
    shader,
    /let g2 = chromaticFrequencyShift \* chromaticFrequencyShift/,
  );
  const sourceStart = shader.indexOf("let intrinsicFlux =");
  const sourceEnd = shader.indexOf("if (!finiteVector(radiance)", sourceStart);
  const source = shader.slice(sourceStart, sourceEnd);
  assert.doesNotMatch(source, /activeWeight|edgeCoverage/);
});

test("dual-disk visible spectrum uses CIE integration and one C2 covering fraction", () => {
  const shader = strongFieldBinaryDualDiskTraceFragmentWGSL;
  const spectralStart = shader.indexOf(
    "fn visibleBlackbodyLinearSrgbPerBolometric(",
  );
  const spectralEnd = shader.indexOf("\n}\n\nfn smootherstep01", spectralStart);
  assert.ok(spectralStart >= 0 && spectralEnd > spectralStart);
  const spectral = shader.slice(spectralStart, spectralEnd);
  assert.match(spectral, /array<vec4<f32>, 15>/);
  assert.match(spectral, /380-780 nm/);
  assert.match(spectral, /linearSrgb = vec3<f32>/);
  assert.match(spectral, /referenceRatio2 \* referenceRatio2/);
  assert.doesNotMatch(spectral, /spectrum \/ luminance|planckChromaticity/);

  const edgeStart = shader.indexOf("fn smootherstep01(");
  const edgeEnd = shader.indexOf("\n}\n\nfn spatialDot", edgeStart);
  const edge = shader.slice(edgeStart, edgeEnd);
  assert.match(edge, /x \* x \* x \* \(x \* \(x \* 6\.0 - 15\.0\) \+ 10\.0\)/);
  assert.match(edge, /innerCoverage \* outerCoverage/);

  const transferStart = shader.indexOf("fn diskTransferAtIntersection(");
  const transferEnd = shader.indexOf("\n}\n\nfn applyDiskIntersection", transferStart);
  const transfer = shader.slice(transferStart, transferEnd);
  assert.equal((transfer.match(/edgeCoverage/g) || []).length, 3);
  assert.match(
    transfer,
    /let opacity = clamp\([\s\S]*activeWeight \* edgeCoverage \* opaqueSurfaceFraction/,
  );
  assert.doesNotMatch(transfer, /tauFace = tauPeak \* activeWeight/);
  assert.doesNotMatch(transfer, /structuredFluxShape[\s\S]*\* activeWeight/);
});

test("dual-disk emissivity texture is bounded, continuous, and transport-only", () => {
  const shader = strongFieldBinaryDualDiskTraceFragmentWGSL;
  const structureStart = shader.indexOf("fn analyticDiskSurfaceStructure(");
  const structureEnd = shader.indexOf("\n}\n\nfn diskTransferAtIntersection", structureStart);
  assert.ok(structureStart >= 0 && structureEnd > structureStart);
  const structure = shader.slice(structureStart, structureEnd);
  assert.match(structure, /let tidal = 0\.16 \* cos/);
  assert.match(structure, /let referenceRadius = \(49\.0 \/ 36\.0\) \* innerRadius/);
  assert.match(structure, /let omegaPeak = sqrt/);
  assert.match(structure, /wrapDiskPatternAngle\(0\.82 \* omegaPeak \* time\)/);
  assert.match(structure, /wrapDiskPatternAngle\(1\.21 \* omegaPeak \* time\)/);
  assert.match(structure, /0\.032 \* sin\(5\.0 \* \(azimuth - phase5\)/);
  assert.match(structure, /0\.024 \* sin\(9\.0 \* \(azimuth - phase9\)/);
  assert.match(structure, /0\.016 \* sin\(14\.0 \* \(azimuth - phase14\)/);
  assert.match(structure, /0\.010 \* sin\(21\.0 \* \(azimuth - phase21\)/);
  assert.match(structure, /0\.008 \* sin\(31\.0 \* \(azimuth - phase31\)/);
  assert.match(structure, /return 1\.0 \+ tidal \+ emissivityTexture/);
  assert.doesNotMatch(structure, /time \* sqrt\([^\n]*radius|clamp\(/);
  assert.doesNotMatch(structure, /position =|momentum =|spatialMetric =|opacity =/);
  assert.match(shader, /deterministic finite-correlation emissivity proxy/i);
});

test("dual-disk photographic mode preserves foreground emission for every ray outcome", () => {
  const shader = strongFieldBinaryDualDiskTraceFragmentWGSL;
  const shadeStart = shader.indexOf("fn shadeResult(");
  const shadeEnd = shader.indexOf("\n}\n\nfn radicalInverse", shadeStart);
  const shade = shader.slice(shadeStart, shadeEnd);
  assert.match(
    shade,
    /RAY_CAPTURED\) \{\s*return result\.diskRadiance;/,
  );
  assert.match(
    shade,
    /RAY_UNRESOLVED\)[\s\S]*return result\.diskRadiance[\s\S]*result\.diskTransmittance[\s\S]*unresolvedLevel \* hatch/,
  );
  assert.match(
    shade,
    /return result\.diskRadiance[\s\S]*result\.diskTransmittance \* sampleEnvironment\(result\.escapeDirection\)[\s\S]*shiftRadiance/,
  );
  const diagnosticPrefix = shade.slice(0, shade.indexOf("if (result.outcome == RAY_CAPTURED)"));
  assert.doesNotMatch(diagnosticPrefix, /diskRadiance|diskTransmittance/);
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

test("WGSL specializes exact inspiral and remnant endpoints", () => {
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /override SPACETIME_PHASE_MODE: i32 = -1/,
  );
  const providerStart = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "fn sampleSpacetime(",
  );
  const providerEnd = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "\n  var g00 = dualConstant(-1.0);",
    providerStart,
  );
  assert.ok(providerStart >= 0 && providerEnd > providerStart);
  const provider = strongFieldBinaryTraceFragmentWGSL.slice(
    providerStart,
    providerEnd,
  );
  const inspiralStart = provider.indexOf("SPACETIME_PHASE_MODE == 0");
  const remnantStart = provider.indexOf(
    "SPACETIME_PHASE_MODE == 1",
    inspiralStart,
  );
  const transitionStart = provider.indexOf("} else {", remnantStart);
  assert.ok(
    inspiralStart >= 0
      && remnantStart > inspiralStart
      && transitionStart > remnantStart,
  );

  const inspiral = provider.slice(inspiralStart, remnantStart);
  assert.match(inspiral, /params\.bodyAPositionMass/);
  assert.match(inspiral, /params\.bodyBPositionMass/);
  assert.doesNotMatch(inspiral, /params\.remnantPositionMass/);
  assert.equal((inspiral.match(/attenuationWeight\(/g) || []).length, 2);

  const remnant = provider.slice(remnantStart, transitionStart);
  assert.match(remnant, /params\.remnantPositionMass/);
  assert.doesNotMatch(remnant, /params\.body[AB]PositionMass/);
  assert.doesNotMatch(remnant, /attenuationWeight\(/);

  const transition = provider.slice(transitionStart);
  assert.match(transition, /params\.bodyAPositionMass/);
  assert.match(transition, /params\.bodyBPositionMass/);
  assert.match(transition, /params\.remnantPositionMass/);
  assert.match(transition, /binaryActive/);
  assert.match(transition, /remnantActive/);
  assert.equal((transition.match(/attenuationWeight\(/g) || []).length, 2);
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
