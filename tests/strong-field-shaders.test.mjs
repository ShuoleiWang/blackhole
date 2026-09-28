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
  // The fallback orbits in the same sense as the WebGPU orbit adapter
  // (angular momentum along +y, phase advancing from +x toward -z).
  assert.match(
    binaryTraceFragmentGLSL,
    /vec3 axis = vec3\(cos\(orbitalPhase\), 0\.0, -sin\(orbitalPhase\)\);/,
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
    "ec466402cb1e6303483db5d85a27cd5b3a1fa8e2dcbcc06169cfcddef1835f46",
  );
  assert.equal(strongFieldBinaryTraceFragmentWGSL.length, 54361);
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
  // Crossings are located on two chords per RK4 step through the third-order
  // continuous-extension midpoint, with momentum interpolated to each hit.
  assert.match(
    shader,
    /let midPosition = stepPosition[\s\S]*accumulateDualDiskEmission\(\s*result,\s*stepPosition,\s*midPosition,[\s\S]*accumulateDualDiskEmission\(\s*result,\s*midPosition,\s*nextPosition,/,
  );
  assert.match(shader, /midX = \(5\.0 \/ 24\.0\) \* derivativeX/);
  assert.match(shader, /let momentumA = mix\(\s*momentumStart,\s*momentumEnd,/);
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
  assert.match(structure, /let referenceRadius = 1\.5918167 \* innerRadius/);
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
    "fn staticObserverFrame(",
    "fn insidePhotonCapture(",
    "photonMargin",
    "failureCaptureMargin",
    "stepDistance",
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
    "MAX_RK4_STEPS: i32 = 192",
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
  // The RK4 step budget comes from the tier; exhausting it outside the
  // capture region stays unresolved.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /if \(stepCount >= maximumSteps\) \{[\s\S]*?failureCaptureMargin < 0\.0[\s\S]*?break;/,
  );
  // Failure capture must cover every term present in the metric: a remnant
  // with w < 1e-4 already breaks the superposition near its ring singularity.
  for (const [weight, term] of [
    ["weightA", "holeA"],
    ["weightB", "holeB"],
    ["weightRemnant", "remnant"],
  ]) {
    assert.match(
      strongFieldBinaryTraceFragmentWGSL,
      new RegExp(
        String.raw`if \(${weight}\.value > 0\.0\) \{\s*failureCaptureMargin = min\(\s*`
          + String.raw`failureCaptureMargin,\s*${term}\.kerrRadius - ${term}\.photonRadius`,
      ),
    );
  }
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
  // Static-observer arriving photon: p^t = (1 - beta.n / alpha_s) / alpha_s.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /let photonTime = \(\s*1\.0 - dot\(observerFrame\.shiftCovariant, initialDirection\) \/ staticLapse\s*\) \/ staticLapse;/,
  );
  // Past-directed flow: position and momentum advance along minus the
  // future-directed Hamiltonian derivatives; exact energy projection keeps
  // the path unchanged because H is homogeneous of degree one in p.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /let derivativeX = -rhs\.velocity;\s*let derivativeP = -rhs\.momentumRate;/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /evalMomentum = evalMomentum\s*\* \(conservedEnergy \/ kinematics\.reducedHamiltonian\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /retryScale = retryScale \* 0\.25/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /dot\(stepPosition, derivativeX\) > 0\.0/,
  );
  // Capture at the innermost photon orbit, with weight-scaled radii.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /2\.0 \* mass \* \(\s*1\.0 \+ cos\(\(2\.0 \/ 3\.0\) \* acos\(-clamp\(length\(safeChi\), 0\.0, 1\.0\)\)\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /let margin = holeA\.kerrRadius - w \* holeA\.photonRadius;/,
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
  const start = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "fn sampleEnvironment(",
  );
  const end = strongFieldBinaryTraceFragmentWGSL.indexOf(
    "\n}\n\nfn viridis(",
    start,
  );
  assert.ok(start >= 0 && end > start);
  const environment = strongFieldBinaryTraceFragmentWGSL.slice(start, end);
  const reconstruction = environment;
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /fn skyQualityPressure\(/,
    "the sky filter must not depend on the Schwarzschild step budget",
  );
  assert.equal(
    (environment.match(/textureSampleLevel\(/g) || []).length,
    5,
    "one centre sample plus four footprint taps are required",
  );
  assert.match(environment, /sourceFootprint/);
  assert.match(environment, /horizontalFov/);
  assert.match(environment, /footprintPressure/);
  assert.match(environment, /clamp\(sourceFootprint \* 0\.72, 0\.50, 3\.0\)/);
  assert.match(environment, /filterWeight = 0\.32 \* footprintPressure/);
  assert.match(environment, /uv \+ vec2<f32>\(radius \* texel\.x, 0\.0\)/);
  assert.match(environment, /uv - vec2<f32>\(0\.0, radius \* texel\.y\)/);
  assert.match(environment, /mix\(centre, filtered, filterWeight\)/);
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

test("RK4 steps scale with the orbit and are bounded by a single ceiling", () => {
  assert.equal(STRONG_FIELD_MAXIMUM_STEP_M, 64);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /params\.sceneStrongIntegrator\.y,[\s\S]*?64\.0/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /stepFraction \* max\(fields\.stepDistance, 0\.0\) \* retryScale/,
  );
  // Classical RK4 weights and a single metric evaluation per iteration.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /\(stepSize \/ 6\.0\) \* \(sumX \+ derivativeX\)/,
  );
  assert.equal(
    strongFieldBinaryTraceFragmentWGSL.match(/sampleSpacetime\(frameTime, /g).length,
    2,
  );
  assert.doesNotMatch(strongFieldBinaryTraceFragmentWGSL, /symplectic|acceptedStepSize/);
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

test("static camera sees the Schwarzschild static-observer sky", () => {
  // Reproduce the shader's static-observer photon at r = 8 M on the +x axis
  // (ingoing Kerr-Schild, M = 1), looking inward at angle psi. Exact results:
  // asymptotic energy E = alpha_s = sqrt(1 - 2M/r) for unit observed energy,
  // and impact parameter b = r sin(psi) / sqrt(1 - 2M/r).
  const mass = 1;
  const radius = 8;
  const fields = schwarzschildKerrSchildAdm(mass, radius);
  const gamma = fields.gammaCovariant;
  const shift = [fields.radialShift, 0, 0];
  const shiftCovariant = matrixVector(gamma, shift);
  const staticLapseSquared = fields.lapse ** 2 - dot(shiftCovariant, shift);
  const staticLapse = Math.sqrt(staticLapseSquared);
  assert.ok(Math.abs(staticLapseSquared - (1 - 2 * mass / radius)) < 1e-14);
  const h = gamma.map((row, i) => row.map(
    (value, j) => value + shiftCovariant[i] * shiftCovariant[j] / staticLapseSquared,
  ));
  const hDot = (a, b) => dot(a, matrixVector(h, b));
  const hNormalize = (value) => value.map((c) => c / Math.sqrt(hDot(value, value)));
  const forward = hNormalize([-1, 0, 0]);
  const right = hNormalize([0, 0, 1]);
  for (const psi of [0.1, 0.42, 0.9]) {
    const n = hNormalize(forward.map((value, i) => value + Math.tan(psi) * right[i]));
    assert.ok(Math.abs(Math.acos(hDot(forward, n)) - psi) < 1e-12);
    const photonTime = (1 - dot(shiftCovariant, n) / staticLapse) / staticLapse;
    const gammaN = matrixVector(gamma, n);
    const p = shiftCovariant.map((value, i) => value * photonTime - gammaN[i]);
    const q = Math.sqrt(dot(p, matrixVector(fields.gammaInverse, p)));
    const energy = fields.lapse * q - dot(shift, p);
    assert.ok(Math.abs(energy - staticLapse) < 1e-12);
    const position = [radius, 0, 0];
    const angularMomentum = Math.hypot(
      position[1] * p[2] - position[2] * p[1],
      position[2] * p[0] - position[0] * p[2],
      position[0] * p[1] - position[1] * p[0],
    );
    const impact = angularMomentum / energy;
    const expected = radius * Math.sin(psi) / Math.sqrt(1 - 2 * mass / radius);
    assert.ok(Math.abs(impact - expected) < 1e-12);
  }
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /fn staticObserverFrame\(fields: ADMFields\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /var momentum = observerFrame\.shiftCovariant \* photonTime\s*- observerFields\.spatialMetric \* initialDirection;/,
  );
});

test("escape directions need no harmonic-gauge tail in ingoing Kerr-Schild", () => {
  // Integrate an arriving photon in ingoing Kerr-Schild Schwarzschild (M = 1)
  // from r = 2e4 M in to a 96 M escape sphere. Its coordinate direction barely
  // changes, while the harmonic-gauge tail (2M/b)(1 - s/r) would add ~0.13 deg.
  const b = 20;
  const escapeRadius = 96;
  const startRadius = 2e4;
  const radiusOf = (x) => Math.hypot(x[1], x[2], x[3]);
  const inverseMetric = (x) => {
    const r = radiusOf(x);
    const H = 1 / r;
    const l = [-1, x[1] / r, x[2] / r, x[3] / r];
    return [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => (
      (i === j ? (i === 0 ? -1 : 1) : 0) - 2 * H * l[i] * l[j]
    )));
  };
  const contract = (g, p) => g.map((row) => row.reduce(
    (sum, value, j) => sum + value * p[j],
    0,
  ));
  const derivative = (y) => {
    const x = y.slice(0, 4);
    const p = y.slice(4);
    const dp = [0, 0, 0, 0];
    for (let axis = 1; axis < 4; axis += 1) {
      const h = 1e-6 * radiusOf(x);
      const plus = x.slice();
      const minus = x.slice();
      plus[axis] += h;
      minus[axis] -= h;
      const gPlus = inverseMetric(plus);
      const gMinus = inverseMetric(minus);
      let sum = 0;
      for (let i = 0; i < 4; i += 1) {
        for (let j = 0; j < 4; j += 1) {
          sum += (gPlus[i][j] - gMinus[i][j]) * p[i] * p[j];
        }
      }
      dp[axis] = -0.25 * sum / h;
    }
    return [...contract(inverseMetric(x), p), ...dp];
  };
  const direction = (y) => {
    const v = derivative(y);
    const d = [v[1] / v[0], v[2] / v[0], v[3] / v[0]];
    const n = Math.hypot(...d);
    return d.map((value) => value / n);
  };
  const x0 = [0, -Math.sqrt(startRadius ** 2 - b ** 2), b, 0];
  const g0 = inverseMetric(x0);
  const spatial = [1, 0, 0];
  const A = g0[0][0];
  const B = 2 * (g0[0][1] * spatial[0] + g0[0][2] * spatial[1] + g0[0][3] * spatial[2]);
  let C = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      C += g0[i + 1][j + 1] * spatial[i] * spatial[j];
    }
  }
  const roots = [1, -1].map((sign) => (-B + sign * Math.sqrt(B * B - 4 * A * C)) / (2 * A));
  const pt = roots.find((root) => contract(g0, [root, ...spatial])[0] > 0);
  let y = [...x0, pt, ...spatial];
  const initial = direction(y);
  while (radiusOf(y) > escapeRadius) {
    const step = Math.min(0.002 * radiusOf(y), radiusOf(y) - escapeRadius + 1e-9);
    const k1 = derivative(y);
    const k2 = derivative(y.map((v, i) => v + 0.5 * step * k1[i]));
    const k3 = derivative(y.map((v, i) => v + 0.5 * step * k2[i]));
    const k4 = derivative(y.map((v, i) => v + step * k3[i]));
    y = y.map((v, i) => v + (step / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }
  const final = direction(y);
  const changeDeg = Math.acos(Math.min(1, dot(initial, final))) * 180 / Math.PI;
  const s = Math.sqrt(escapeRadius ** 2 - b ** 2);
  const harmonicTailDeg = (2 / b) * (1 - s / escapeRadius) * 180 / Math.PI;
  assert.ok(changeDeg < 0.02, `KS far-field change ${changeDeg} deg`);
  assert.ok(harmonicTailDeg > 10 * changeDeg);
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /asymptoticEscapeDirection|remainingFraction/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /result\.escapeDirection = safeNormalize\(derivativeX\)/,
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
