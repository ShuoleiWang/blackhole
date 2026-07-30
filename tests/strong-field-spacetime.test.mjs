import assert from "node:assert/strict";
import test from "node:test";

import {
  STRONG_FIELD_UNIFORM_ABI,
  createPnEobOrbitAdapter,
  createStrongFieldSpacetimeProvider,
  evaluateKerrSchild3p1,
  hamiltonianDerivatives,
  nullHamiltonian,
  nullHamiltonianResidual,
} from "../src/strong-field-spacetime.js";

function close(actual, expected, tolerance, message = "") {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message} expected ${expected}, received ${actual}`,
  );
}

function finiteDeep(value) {
  if (typeof value === "number") {
    assert.ok(Number.isFinite(value));
  } else if (Array.isArray(value)) {
    value.forEach(finiteDeep);
  }
}

function blackHole(
  id,
  massM,
  positionM,
  {
    velocityC = [0, 0, 0],
    dimensionlessSpin = [0, 0, 0],
  } = {},
) {
  return {
    id,
    massM,
    positionM,
    velocityC,
    dimensionlessSpin,
  };
}

function adapterFor(sample) {
  return createPnEobOrbitAdapter({
    dynamicsModel: "4PN/EOB-compatible test trajectory",
    coordinateFrame: "asymptotically-inertial-kerr-schild-com",
    source: "deterministic unit-test orbit",
    usesSxsGaugeCentroids: false,
    sample,
  });
}

function binarySample({
  separationM = 20,
  velocity = 0,
  blend = 0,
  spinA = [0, 0, 0],
  spinB = [0, 0, 0],
  remnantSpin = [0, 0, 0.68],
} = {}) {
  return {
    bodies: [
      blackHole("A", 0.5, [-separationM / 2, 0, 0], {
        velocityC: [0, 0, velocity],
        dimensionlessSpin: spinA,
      }),
      blackHole("B", 0.5, [separationM / 2, 0, 0], {
        velocityC: [0, 0, -velocity],
        dimensionlessSpin: spinB,
      }),
    ],
    remnant: blackHole("R", 0.9516, [0, 0, 0], {
      dimensionlessSpin: remnantSpin,
    }),
    mergerBlend: blend,
  };
}

test("Minkowski limit has unit lapse and exact null Hamiltonian flow", () => {
  const fields = evaluateKerrSchild3p1({
    massM: 0,
    positionM: [7, -3, 2],
  });
  close(fields.lapse, 1, 0);
  assert.deepEqual(fields.shift, [0, 0, 0]);
  assert.deepEqual(fields.inverseSpatialMetric, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  const momentum = [2, -3, 6];
  close(nullHamiltonian(fields, momentum), 7, 1e-14);
  const residual = nullHamiltonianResidual(fields, momentum);
  close(residual.raw, 0, 1e-14);
  assert.equal(residual.futureDirected, true);

  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample({
      separationM: 1e9,
    })),
    attenuation: { enabled: false },
  });
  // The dedicated Minkowski result above is exact.  The derivative test uses a
  // far-field binary so it exercises the provider and tends to flat flow.
  const flow = hamiltonianDerivatives(
    provider,
    0,
    [0, 1e12, 0],
    momentum,
  );
  close(flow.dxdt[0], 2 / 7, 2e-9);
  close(flow.dxdt[1], -3 / 7, 2e-9);
  close(flow.dxdt[2], 6 / 7, 2e-9);
  assert.ok(Math.hypot(...flow.dpdt) < 1e-16);
});
test("single Schwarzschild Kerr-Schild 3+1 fields match closed form", () => {
  const radius = 10;
  const mass = 1;
  const fields = evaluateKerrSchild3p1({
    massM: mass,
    positionM: [radius, 0, 0],
  });
  const twoH = 2 * mass / radius;
  close(fields.lapse, 1 / Math.sqrt(1 + twoH), 2e-15);
  close(fields.shift[0], twoH / (1 + twoH), 2e-15);
  close(fields.shift[1], 0, 0);
  close(fields.inverseSpatialMetric[0][0], 1 / (1 + twoH), 2e-15);
  close(fields.inverseSpatialMetric[1][1], 1, 2e-15);
  close(fields.covariantMetric[0][0], -1 + twoH, 2e-15);
  close(fields.horizonRadiusM, 2 * mass, 0);
  assert.equal(fields.regularized, false);
});

test("positive Kerr spin produces the expected frame-dragging sign", () => {
  const positive = evaluateKerrSchild3p1({
    massM: 1,
    positionM: [8, 0, 0],
    dimensionlessSpin: [0, 0, 0.7],
  });
  const negative = evaluateKerrSchild3p1({
    massM: 1,
    positionM: [8, 0, 0],
    dimensionlessSpin: [0, 0, -0.7],
  });
  assert.ok(positive.covariantMetric[0][2] < 0);
  assert.ok(negative.covariantMetric[0][2] > 0);
  close(
    positive.covariantMetric[0][2],
    -negative.covariantMetric[0][2],
    2e-15,
  );
  close(positive.lapse, negative.lapse, 2e-15);
});

test("wide-separation superposition recovers the weak-field monopole", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample({
      separationM: 100,
    })),
    attenuation: { enabled: false },
  });
  const position = [0, 1000, 0];
  const fields = provider.evaluate(0, position);
  const distance = Math.hypot(50, 1000);
  const expectedPerturbation = 2 / distance;
  close(
    fields.covariantMetric[0][0] + 1,
    expectedPerturbation,
    2e-15,
  );
  close(
    fields.covariantMetric[0][0] + 1,
    2 / 1000,
    3e-6,
  );
  assert.equal(fields.constraintSolved, false);
  assert.match(fields.scientificStatus, /approximate metric; not NR/);
});

test("companion attenuation isolates each singular neighborhood", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample({
      separationM: 20,
    })),
  });
  const atBodyA = provider.evaluate(0, [-10, 0, 1e-3]);
  assert.ok(atBodyA.diagnostics.companionAttenuation[0] > 0.99);
  assert.ok(atBodyA.diagnostics.companionAttenuation[1] < 1e-12);
  assert.equal(atBodyA.regularized, false);
});

test("inspiral-to-remnant blend is C2 and lands on exact Kerr", () => {
  const adapter = adapterFor((timeM) => binarySample({
    separationM: 8,
    velocity: 0.2,
    blend: Math.min(1, Math.max(0, (timeM + 1) / 2)),
  }));
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapter,
    attenuation: { enabled: false },
  });
  const position = [0, 0, 12];
  const before = provider.evaluate(-1 - 1e-4, position);
  const edge = provider.evaluate(-1, position);
  const after = provider.evaluate(-1 + 1e-4, position);
  close(before.transitionWeight, 0, 0);
  close(edge.transitionWeight, 0, 0);
  assert.ok(after.transitionWeight < 2e-11);
  assert.ok(
    Math.abs(after.lapse - edge.lapse)
      < Math.abs(provider.evaluate(0, position).lapse - edge.lapse) * 1e-8,
  );

  const final = provider.evaluate(1, position);
  const later = provider.evaluate(2, position);
  const exact = evaluateKerrSchild3p1({
    massM: 0.9516,
    positionM: position,
    dimensionlessSpin: [0, 0, 0.68],
  });
  close(final.transitionWeight, 1, 0);
  close(final.lapse, exact.lapse, 3e-15);
  close(later.lapse, exact.lapse, 3e-15);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      close(
        final.covariantMetric[row][column],
        exact.covariantMetric[row][column],
        3e-15,
      );
    }
  }
});

test("regularization is finite and explicitly flagged near Kerr rings", () => {
  const samples = [
    evaluateKerrSchild3p1({
      massM: 1,
      positionM: [0, 0, 0],
    }),
    evaluateKerrSchild3p1({
      massM: 1,
      positionM: [0.9, 0, 0],
      dimensionlessSpin: [0, 0, 0.9],
    }),
    evaluateKerrSchild3p1({
      massM: 1,
      positionM: [0, 0, 1e-12],
      dimensionlessSpin: [0, 0, 0.99],
    }),
  ];
  for (const fields of samples) {
    assert.equal(fields.regularized, true);
    finiteDeep(fields.covariantMetric);
    finiteDeep(fields.inverseMetric);
    assert.ok(fields.lapse > 0);
  }
});

test("null Hamiltonian construction remains null across the strong field", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample({
      separationM: 12,
      velocity: 0.17,
      blend: 0.35,
      spinA: [0, 0, 0.3],
      spinB: [0, 0, -0.2],
    })),
  });
  const points = [
    [0, 0, 30],
    [-6, 0.5, 3],
    [6, -0.5, 3],
    [0, 12, 0],
  ];
  const momenta = [
    [1, 0.2, -0.3],
    [-0.4, 0.7, 0.1],
    [0.2, -0.1, 1],
  ];
  for (const point of points) {
    const fields = provider.evaluate(0, point);
    for (const momentum of momenta) {
      const residual = nullHamiltonianResidual(fields, momentum);
      assert.ok(residual.normalized < 2e-14);
      assert.equal(residual.futureDirected, true);
    }
  }
});

test("Hamiltonian spatial derivatives agree with an independent difference", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor((timeM) => binarySample({
      separationM: 18,
      velocity: 0.12,
      blend: 0.2 + 0.01 * Math.sin(timeM),
    })),
  });
  const timeM = 0.4;
  const position = [1.5, 8, 13];
  const momentum = [0.7, -0.2, 1.1];
  const flow = hamiltonianDerivatives(
    provider,
    timeM,
    position,
    momentum,
  );
  finiteDeep(flow.dxdt);
  finiteDeep(flow.dpdt);
  assert.ok(Number.isFinite(flow.dHdt));
  assert.ok(flow.residual.normalized < 2e-14);
  const step = 1e-4;
  const lower = position.slice();
  const upper = position.slice();
  lower[1] -= step;
  upper[1] += step;
  const independent = -(
    nullHamiltonian(provider.evaluate(timeM, upper), momentum)
    - nullHamiltonian(provider.evaluate(timeM, lower), momentum)
  ) / (2 * step);
  close(flow.dpdt[1], independent, 2e-8);
});

test("orbit contract rejects SXS gauge centroids as physical positions", () => {
  assert.throws(
    () => createPnEobOrbitAdapter({
      dynamicsModel: "EOB",
      source: "SXS horizon coordinate centers",
      usesSxsGaugeCentroids: true,
      sample: () => binarySample(),
    }),
    /SXS horizon-centroid coordinates cannot be used/,
  );
  assert.throws(
    () => createPnEobOrbitAdapter({
      dynamicsModel: "unlabelled interpolation",
      source: "unknown",
      usesSxsGaugeCentroids: false,
      sample: () => binarySample(),
    }),
    /PN or EOB/,
  );
});

test("GPU parameter packet is complete, finite, and stable", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample({
      separationM: 17,
      velocity: 0.11,
      blend: 0.4,
    })),
  });
  const frame = provider.frameAt(12.5);
  assert.ok(frame.uniforms instanceof Float32Array);
  assert.equal(frame.uniforms.length, STRONG_FIELD_UNIFORM_ABI.floatCount);
  assert.equal(frame.uniforms.byteLength, STRONG_FIELD_UNIFORM_ABI.byteLength);
  finiteDeep([...frame.uniforms]);
  close(frame.uniforms[0], 12.5, 0);
  close(frame.uniforms[3], provider.regularization.radiusFraction, 1e-11);
  close(frame.uniforms[43], provider.regularization.maxKerrSchildH, 0);
});

test("invalid metric domains fail closed as unresolved, never fake sky", () => {
  const provider = createStrongFieldSpacetimeProvider({
    orbitAdapter: adapterFor(() => binarySample()),
  });
  const frame = provider.frameAt(0);
  const result = frame.evaluateOrUnresolved([Number.NaN, 0, 0]);
  assert.equal(result.outcome, "unresolved");
  assert.equal(result.fields, null);
  assert.match(result.reason, /finite number/);
});
