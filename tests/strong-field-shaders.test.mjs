import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { binaryTraceFragmentGLSL } from "../src/binary-shaders.js";
import {
  STRONG_FIELD_UNIFORM_ABI,
  createPnEobOrbitAdapter,
  createStrongFieldSpacetimeProvider,
  evaluateKerrSchild3p1,
  nullHamiltonian,
} from "../src/strong-field-spacetime.js";
import {
  STRONG_FIELD_DIAGNOSTIC_MODES,
  STRONG_FIELD_ACCRETION_UNIFORM_FLOATS,
  STRONG_FIELD_ACCRETION_UNIFORM_LAYOUT,
  STRONG_FIELD_ACCRETION_UNIFORM_TAIL_FLOATS,
  STRONG_FIELD_MAXIMUM_STEP_M,
  STRONG_FIELD_OUTCOMES,
  STRONG_FIELD_SPARSE_STRIDE,
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
      ({ id, constants }) => [
        id,
        constants.SPACETIME_PHASE_MODE,
        constants.BINARY_SPIN_MODE,
      ],
    ),
    [
      ["binary", 0, 0],
      ["binary-spinning", 0, 1],
      ["transition", 2, 0],
      ["transition-spinning", 2, 1],
      ["remnant", 1, 0],
    ],
  );
  const selectPhase = strongFieldBinaryShaderBundle.wgsl
    .selectTraceSpecialization;
  const packet = (blend, spinA = [0, 0, 0], spinB = [0, 0, 0]) => {
    const uniforms = new Float32Array(44);
    uniforms[1] = blend;
    uniforms.set(spinA, 12);
    uniforms.set(spinB, 24);
    return { sceneStrongFieldUniforms: uniforms };
  };
  assert.equal(selectPhase(packet(0)), "binary");
  assert.equal(selectPhase(packet(0.5)), "transition");
  assert.equal(selectPhase(packet(1)), "remnant");
  // SXS:BBH:0001 spins (~1e-9) keep the non-spinning specialization.
  assert.equal(selectPhase(packet(0, [7e-10, 7e-10, 0])), "binary");
  assert.equal(selectPhase(packet(0, [0, 0, 0.3])), "binary-spinning");
  assert.equal(selectPhase(packet(0.5, [0, 0, 0], [0.1, 0, 0])), "transition-spinning");
  assert.equal(selectPhase(packet(1, [0, 0, 0.3])), "remnant");
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
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.wgsl.traceSpecializations,
    strongFieldBinaryShaderBundle.wgsl.traceSpecializations,
  );
  assert.equal(
    strongFieldBinaryDualDiskShaderBundle.wgsl.selectTraceSpecialization,
    strongFieldBinaryShaderBundle.wgsl.selectTraceSpecialization,
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
    "9eb37a98dbf9234682513d1cb562a1d1a85ecfd658d7f8f80735cd877e7df93e",
  );
  assert.equal(strongFieldBinaryTraceFragmentWGSL.length, 61859);
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
    "struct KerrSchildTerm",
    "fn kerrSchildTerm(",
    "fn termJacobianTranspose(",
    "fn spacetimeTerms(",
    "fn lowRank1(",
    "fn lowRank2(",
    "fn lowRank3(",
    "fn evaluateGeodesic(",
    "fn metricValuesAt(",
    "struct GeodesicSample",
    "override BINARY_SPIN_MODE: i32 = 1;",
    "bodyAPositionMass",
    "bodyAVelocityActive",
    "bodyASpin",
    "fn companionAttenuation(",
    "observerCameraDirection",
    "fn staticObserverFrame(",
    "fn insidePhotonCapture(",
    "photonMargin",
    "failureCaptureMargin",
    "stepDistance",
    "binaryActive",
    "Reduced 3+1 null Hamiltonian",
    "RAY_CAPTURED",
    "RAY_ESCAPED",
    "RAY_UNRESOLVED",
    "escapeDirection",
    "frequencyShift",
    "lookback",
    "hamiltonianResidual",
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
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /if \(weight > 0\.0\) \{\s*\(\*sample\)\.failureCaptureMargin = min\(\s*\(\*sample\)\.failureCaptureMargin,\s*term\.kerrRadius - photonRadius/,
  );
  for (const [term, weight, radius] of [["a", "weightA", "A"], ["b", "weightB", "B"], ["r", "weightR", "R"]]) {
    assert.match(
      strongFieldBinaryTraceFragmentWGSL,
      new RegExp(String.raw`addTermGeometry\(&sample, terms\.${term}, terms\.${weight}, radii\.horizon${radius}, radii\.photon${radius}\);`),
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
    /let derivativeX = -sample\.velocity;\s*let derivativeP = -sample\.momentumRate \* momentumScale;/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /momentumScale = conservedEnergy \/ sample\.hamiltonian;\s*evalMomentum = evalMomentum \* momentumScale;/,
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
    /2\.0 \* mass \* \(1\.0 \+ cos\(\(2\.0 \/ 3\.0\) \* acos\(-chi\)\)\)/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /let margin = term\.kerrRadius - w \* photonRadius;/,
  );
});

test("WGSL specializes exact inspiral and remnant endpoints", () => {
  const wgsl = strongFieldBinaryTraceFragmentWGSL;
  assert.match(wgsl, /override SPACETIME_PHASE_MODE: i32 = -1/);
  const slice = (name, next) => {
    const start = wgsl.indexOf(`fn ${name}(`);
    const end = wgsl.indexOf(`fn ${next}(`, start);
    assert.ok(start >= 0 && end > start, name);
    return wgsl.slice(start, end);
  };
  const phase = slice("spacetimePhase", "spacetimeTerms");
  assert.match(phase, /SPACETIME_PHASE_MODE == 1 \|\| \(SPACETIME_PHASE_MODE < 0 && blend == 1\.0\)/);
  assert.match(phase, /SPACETIME_PHASE_MODE == 0 \|\| \(SPACETIME_PHASE_MODE < 0 && blend == 0\.0\)/);

  const terms = slice("spacetimeTerms", "nullEnergy");
  const remnantOnly = terms.slice(terms.indexOf("if (phase == 1) {"), terms.indexOf("return terms;"));
  assert.match(remnantOnly, /params\.remnantPositionMass/);
  assert.doesNotMatch(remnantOnly, /params\.body[AB]PositionMass/);
  const binary = terms.slice(terms.indexOf("return terms;") + 1);
  assert.match(binary, /params\.bodyAPositionMass/);
  assert.match(binary, /params\.bodyBPositionMass/);
  assert.match(binary, /if \(phase == 2\) \{[\s\S]*params\.remnantPositionMass/);
  assert.match(binary, /binaryActive/);
  // Non-spinning bodies compile out the Kerr branch; the remnant keeps it.
  assert.equal((binary.match(/BINARY_SPIN_MODE != 0/g) || []).length, 2);
  assert.equal((terms.match(/params\.remnantVelocityActive\.w[^\n]*,\n\s*true/g) || []).length, 2);
  assert.equal((binary.match(/companionAttenuation\(/g) || []).length, 2);

  const evaluate = slice("evaluateGeodesic", "termOuter");
  assert.match(evaluate, /if \(phase == 1\) \{\s*flow = lowRank1\(terms\.r, momentum\);/);
  assert.match(evaluate, /else if \(phase == 0\) \{\s*flow = lowRank2\(terms\.a, terms\.b, momentum\);/);
  assert.match(evaluate, /flow = lowRank3\(terms\.a, terms\.b, terms\.r, momentum\);/);
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
    /effectiveFraction \* max\(sample\.stepDistance, 0\.0\) \* retryScale/,
  );
  // Larger steps only for outbound rays beyond 30 M.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /let outbound = radius > FAR_FIELD_RADIUS_M\s*&& dot\(stepPosition, derivativeX\) > 0\.0;/,
  );
  assert.match(strongFieldBinaryTraceFragmentWGSL, /const FAR_FIELD_RADIUS_M: f32 = 30\.0;/);
  assert.match(strongFieldBinaryTraceFragmentWGSL, /const FAR_FIELD_GROWTH: f32 = 2\.0;/);
  // Classical RK4 weights and a single metric evaluation per iteration.
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /\(stepSize \/ 6\.0\) \* \(sumX \+ derivativeX\)/,
  );
  assert.equal(
    strongFieldBinaryTraceFragmentWGSL.match(/evaluateGeodesic\(/g).length,
    2,
    "one definition and one call site in the RK4 loop",
  );
  assert.doesNotMatch(strongFieldBinaryTraceFragmentWGSL, /symplectic|acceptedStepSize/);
});

test("sparse tracing interpolates only provably smooth sky and fails closed", () => {
  const coarse = strongFieldBinaryShaderBundle.wgsl.coarse;
  assert.deepEqual({ ...coarse }, {
    entryPoint: "fsCoarse",
    format: "rgba32float",
    stride: STRONG_FIELD_SPARSE_STRIDE,
    binding: 3,
  });
  assert.equal(strongFieldBinaryDualDiskShaderBundle.wgsl.coarse, coarse);
  assert.equal(STRONG_FIELD_SPARSE_STRIDE, 4);
  for (const wgsl of [
    strongFieldBinaryTraceFragmentWGSL,
    strongFieldBinaryDualDiskTraceFragmentWGSL,
  ]) {
    assert.match(
      wgsl,
      /@group\(0\) @binding\(3\) var coarseField: texture_2d<f32>;/,
    );
    assert.match(wgsl, /const SPARSE_STRIDE: f32 = 4\.0;/);
    assert.match(wgsl, /const SPARSE_CURVATURE_PIXELS: f32 = 0\.25;/);
    // Coarse node (i, j), stored at texel (i + 1, j + 1), traces pixel
    // (i S, j S) and stores direction * frequency shift plus outcome code.
    const coarseEntry = wgsl.slice(
      wgsl.indexOf("fn fsCoarse("),
      wgsl.indexOf("struct SparseSample"),
    );
    assert.match(coarseEntry, /floor\(input\.position\.xy\) - vec2<f32>\(1\.0\)/);
    assert.match(coarseEntry, /node \* SPARSE_STRIDE \+ vec2<f32>\(0\.5\)/);
    assert.match(
      coarseEntry,
      /traceStrongFieldRay\([\s\S]*SPARSE_FOOTPRINT_SPACINGS \* SPARSE_STRIDE \* pixelAngle/,
    );
    assert.match(
      coarseEntry,
      /vec4<f32>\(result\.escapeDirection \* result\.frequencyShift, code\)/,
    );
    // All twelve stencil nodes must have escaped (code 2), and the bilinear
    // error bound must stay below SPARSE_CURVATURE_PIXELS.
    const interpolation = wgsl.slice(
      wgsl.indexOf("fn sparseInterpolation("),
      wgsl.indexOf("@fragment\nfn fsMain"),
    );
    assert.match(
      interpolation,
      /abs\(corners - vec4<f32>\(2\.0\)\) > vec4<f32>\(0\.25\)/,
    );
    assert.match(interpolation, /abs\(neighbours\[index\] - 2\.0\) > 0\.25/);
    assert.equal((interpolation.match(/coarseNode\(base/g) || []).length, 12);
    assert.match(
      interpolation,
      /0\.125 \* curvature > SPARSE_CURVATURE_PIXELS \* pixelAngle \* length\(c00\.xyz\)/,
    );
    // Only unjittered photographic frames that requested the declared
    // stride use the field; every other pixel is traced as before.
    const main = wgsl.slice(wgsl.indexOf("@fragment\nfn fsMain"));
    assert.match(
      main,
      /params\.sceneStrongDiagnostics\.w == SPARSE_STRIDE\s*&& i32\(round\(params\.renderControls\.z\)\) == 0\s*&& params\.sceneStrongQuality\.w > 0\.5/,
    );
    assert.match(main, /let result = traceStrongField\(screen, tanHalfFov\);/);
    assert.match(
      wgsl,
      /fn traceStrongField\(screen: vec2<f32>, tanHalfFov: f32\) -> RayResult \{\s*return traceStrongFieldRay\(screen, tanHalfFov, 0\.0\);/,
    );
  }
  // A dual-disk coarse node is also flagged when its beam (the footprint
  // around its ray) passes a disk annulus, so a disk image thinner than the
  // node spacing, e.g. edge-on, is traced rather than interpolated away.
  const dual = strongFieldBinaryDualDiskTraceFragmentWGSL;
  assert.match(dual, /const SPARSE_FOOTPRINT_SPACINGS: f32 = 2\.0;/);
  assert.match(
    dual,
    /\|\| result\.diskProximity > 0\.5\s*\)\s*\{\s*code = code \+ 4\.0;/,
  );
  assert.match(dual, /let footprint = footprintAngle \* \(lookback \+ stepSize\);/);
  assert.match(
    dual,
    /chordNearDisks\(stepPosition, midPosition, footprint\)\s*\|\| chordNearDisks\(midPosition, nextPosition, footprint\)/,
  );
  const nearDisk = dual.slice(
    dual.indexOf("fn chordNearDisk("),
    dual.indexOf("fn chordNearDisks("),
  );
  assert.match(nearDisk, /return true;/);
  assert.match(
    nearDisk,
    /nearest <= outerRadius \+ footprint\s*&& farthest >= innerRadius - footprint/,
  );
  assert.doesNotMatch(
    strongFieldBinaryTraceFragmentWGSL,
    /diskProximity|chordNearDisk/,
  );
});

test("rays hovering at a trapped surface inside a photon orbit are captured", () => {
  for (const wgsl of [
    strongFieldBinaryTraceFragmentWGSL,
    strongFieldBinaryDualDiskTraceFragmentWGSL,
  ]) {
    // Where a hole's Kerr-Schild field 2H is >= 2/3, escaping past-directed
    // rays move at coordinate speed >= 1/sqrt(3) (Schwarzschild) or 0.55
    // (Kerr to chi = 0.95); the threshold keeps a factor ~2.7 below that.
    assert.match(wgsl, /const HOVER_CAPTURE_SPEED: f32 = 0\.2;/);
    assert.match(wgsl, /const HOVER_CAPTURE_FIELD: f32 = 2\.0 \/ 3\.0;/);
    // Fields add along the terms' null directions, so they cancel at the
    // saddle between two holes.
    assert.match(
      wgsl,
      /sample\.alignedField = length\(\s*terms\.a\.c \* terms\.a\.n \+ terms\.b\.c \* terms\.b\.n \+ terms\.r\.c \* terms\.r\.n\s*\);/,
    );
    const capture = wgsl.slice(
      wgsl.indexOf("fn insidePhotonCapture("),
      wgsl.indexOf("fn traceStrongField("),
    );
    assert.match(
      capture,
      /let photonOrbitCapture = sample\.photonMargin < 0\.0\s*&& dot\(sample\.photonRadialGradient, backwardVelocity\) < 0\.0;/,
    );
    assert.match(
      capture,
      /let hoverCapture = sample\.failureCaptureMargin < 0\.0\s*&& sample\.alignedField >= HOVER_CAPTURE_FIELD\s*&& length\(backwardVelocity\) < HOVER_CAPTURE_SPEED;/,
    );
    assert.match(capture, /return photonOrbitCapture \|\| hoverCapture;/);
    // The test runs on the start-of-step sample with the backward velocity.
    assert.match(wgsl, /if \(insidePhotonCapture\(sample, derivativeX\)\) \{/);
  }
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
    /fn staticObserverFrame\(values: MetricValues\)/,
  );
  assert.match(strongFieldBinaryTraceFragmentWGSL, /let lapseSquared = -values\.gtt;/);
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /var momentum = observerFrame\.shiftCovariant \* photonTime\s*- observerMetric\.spatialMetric \* initialDirection;/,
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
  // The frozen WGSL terms are unboosted: no velocity enters the metric.
  const termStart = strongFieldBinaryTraceFragmentWGSL.indexOf("fn kerrSchildTerm(");
  const termSignature = strongFieldBinaryTraceFragmentWGSL.slice(
    termStart,
    strongFieldBinaryTraceFragmentWGSL.indexOf("{", termStart),
  );
  assert.doesNotMatch(termSignature, /velocity/i);
  assert.doesNotMatch(strongFieldBinaryTraceFragmentWGSL, /boostGamma|velocityDotDirection/);
});

test("low-rank Kerr-Schild Hamiltonian reproduces the 3+1 oracle", () => {
  // Float64 mirror of the WGSL provider: per-term c = 2 w H, unit n and the
  // closed-form Jacobian products, the capacitance system (I + N C) z = r, the
  // quadratic for E(x,p), and the flow dx/dt = dH/dp, dp/dt = -dH/dx. It is
  // checked against the CPU oracle (H directly, its derivatives by central
  // differences) for one, two and three terms, including spinning holes.
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const scale = (a, k) => a.map((x) => x * k);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const term = (body, weight, position) => {
    const x = sub(position, body.positionM);
    const a = scale(body.dimensionlessSpin, body.massM);
    const s = dot(a, x);
    const reduced = dot(x, x) - dot(a, a);
    const r2 = 0.5 * (reduced + Math.sqrt(reduced * reduced + 4 * s * s));
    const r = Math.sqrt(r2);
    const W = r2 * r2 + s * s;
    const radiusGradient = scale(add(scale(x, r2), scale(a, s)), r / W);
    const H = body.massM * r * r2 / W;
    const hGradient = scale(
      sub(scale(radiusGradient, 3 * s * s - r2 * r2), scale(a, 2 * r * s)),
      body.massM * r2 / (W * W),
    );
    const inverseNormalization = 1 / (r2 + dot(a, a));
    const n = scale(add(add(scale(x, r), cross(x, a)), scale(a, s / r)), inverseNormalization);
    const w = sub(sub(x, scale(a, s / r2)), scale(n, 2 * r));
    const jacobianTranspose = (v) => scale(
      add(add(add(scale(radiusGradient, dot(w, v)), scale(v, r)), cross(a, v)), scale(a, dot(a, v) / r)),
      inverseNormalization,
    );
    return { c: 2 * weight * H, cGradient: scale(hGradient, 2 * weight), n, jacobianTranspose };
  };
  const solve = (A, b) => {
    if (b.length === 1) return [b[0] / A[0][0]];
    const det3 = (M) => (M.length === 2
      ? M[0][0] * M[1][1] - M[0][1] * M[1][0]
      : M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]));
    const det = det3(A);
    return b.map((_, k) => det3(A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / det);
  };
  const lowRank = (terms, p) => {
    const c = terms.map((t) => t.c);
    const A = terms.map((ta, i) => terms.map((tb, j) => (i === j ? 1 : (dot(ta.n, tb.n) - 1) * c[j])));
    const v = terms.map((t) => dot(t.n, p));
    const ones = solve(A, terms.map(() => 1));
    const projected = solve(A, v);
    const onePlusK = 1 + ones.reduce((sum, z, i) => sum + c[i] * z, 0);
    const m = projected.reduce((sum, z, i) => sum + c[i] * z, 0);
    const pq = dot(p, p) - projected.reduce((sum, z, i) => sum + c[i] * z * v[i], 0);
    const root = Math.sqrt(m * m + onePlusK * pq);
    const energy = m > 0 ? pq / (root + m) : (root - m) / onePlusK;
    const z = ones.map((o, i) => energy * o + projected[i]);
    const y = z.map((zi, i) => c[i] * zi);
    let velocity = p.slice();
    let gradient = [0, 0, 0];
    terms.forEach((t, i) => {
      velocity = sub(velocity, scale(t.n, y[i]));
      gradient = add(gradient, add(scale(t.jacobianTranspose(p), 2 * y[i]), scale(t.cGradient, z[i] * z[i])));
      for (let j = i + 1; j < terms.length; j += 1) {
        const coupling = add(t.jacobianTranspose(terms[j].n), terms[j].jacobianTranspose(t.n));
        gradient = sub(gradient, scale(coupling, 2 * y[i] * y[j]));
      }
    });
    return { energy, velocity: scale(velocity, 1 / root), momentumRate: scale(gradient, 0.5 / root) };
  };
  const hole = (id, massM, positionM, dimensionlessSpin = [0, 0, 0]) => ({
    id, massM, positionM, velocityC: [0, 0, 0], dimensionlessSpin,
  });
  const cases = [
    { blend: 0, x: [3.1, -2.4, 5.7], spinA: [0, 0, 0] },
    { blend: 0, x: [-1.2, 0.8, 2.2], spinA: [0.2, -0.3, 0.4] },
    { blend: 0.37, x: [0.9, 1.3, -2.1], spinA: [0, 0, 0] },
    { blend: 1, x: [2.4, 0.3, -1.8], spinA: [0, 0, 0] },
  ];
  for (const { blend, x, spinA } of cases) {
    const bodies = [hole("A", 0.5, [2.6, 0, -0.9], spinA), hole("B", 0.5, [-2.6, 0, 0.9])];
    const remnant = hole("R", 0.9516, [0, 0, 0], [0, 0.686, 0]);
    const frame = createStrongFieldSpacetimeProvider({
      orbitAdapter: createPnEobOrbitAdapter({
        dynamicsModel: "4PN/EOB-compatible low-rank oracle check",
        coordinateFrame: "asymptotically-inertial-kerr-schild-com",
        source: "deterministic shader contract fixture",
        usesSxsGaugeCentroids: false,
        sample: () => ({ bodies, remnant, mergerBlend: blend }),
      }),
    }).frameAt(0);
    const fieldsAt = (point) => {
      const sample = frame.evaluateOrUnresolved(point);
      assert.equal(sample.outcome, "valid");
      return sample.fields;
    };
    const p = [0.31, -0.72, 0.45];
    const fields = fieldsAt(x);
    const w = fields.transitionWeight;
    const terms = [];
    if (w < 1) terms.push(term(bodies[0], 1 - w, x), term(bodies[1], 1 - w, x));
    if (w > 0) terms.push(term(remnant, w, x));
    const flow = lowRank(terms, p);
    const H = nullHamiltonian(fields, p);
    assert.ok(Math.abs(flow.energy / H - 1) < 1e-12, `H at blend ${blend}`);
    const h = 1e-6;
    for (let j = 0; j < 3; j += 1) {
      const xp = x.slice(); xp[j] += h;
      const xm = x.slice(); xm[j] -= h;
      const pp = p.slice(); pp[j] += h;
      const pm = p.slice(); pm[j] -= h;
      const dHdx = (nullHamiltonian(fieldsAt(xp), p) - nullHamiltonian(fieldsAt(xm), p)) / (2 * h);
      const dHdp = (nullHamiltonian(fields, pp) - nullHamiltonian(fields, pm)) / (2 * h);
      assert.ok(Math.abs(flow.velocity[j] - dHdp) < 1e-7, `dx/dt at blend ${blend}`);
      assert.ok(Math.abs(flow.momentumRate[j] + dHdx) < 1e-6, `dp/dt at blend ${blend}`);
    }
  }
});

