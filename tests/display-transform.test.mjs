import assert from "node:assert/strict";
import test from "node:test";

import { postFragmentGLSL, postFragmentWGSL } from "../src/shaders.js";
import {
  strongFieldBinaryDualDiskShaderBundle,
  strongFieldBinaryShaderBundle,
  strongFieldBinaryTraceFragmentWGSL,
} from "../src/strong-field-shaders.js";
import { WebGLRenderer } from "../src/webgl-renderer.js";
import { WebGPURenderer } from "../src/webgpu-renderer.js";

function encodeSrgb(value) {
  return value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
}

function decodeSrgb(value) {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function frame(overrides = {}) {
  return {
    time: 0,
    massSolar: 1,
    accretion: 0,
    exposure: 3,
    mode: 2,
    steps: 160,
    cameraPos: [0, 0, 40],
    cameraRadius: 40,
    forward: [0, 0, -1],
    fov: 0.7,
    right: [1, 0, 0],
    skyRotation: 0,
    up: [0, 1, 0],
    diskOuterRadius: 18,
    renderScale: 1,
    bloom: 0,
    motion: 1,
    frame: 7,
    observerVelocity: [0, 0, 0],
    observerBeta: 0,
    ...overrides,
  };
}

test("post passes diagnostic false colour through without exposure or tone mapping", () => {
  const body = postFragmentWGSL.slice(postFragmentWGSL.indexOf("fn fsMain("));
  const branchStart = body.indexOf("if (params.postDisplayFrame.z > 0.5) {");
  const exposureStart = body.indexOf("sampleScene(input.uv) * exposure");
  assert.ok(branchStart > 0 && exposureStart > branchStart);
  const branch = body.slice(branchStart, body.indexOf("\n  }\n", branchStart));
  assert.match(branch, /linearSrgbToDisplayP3\(falseColour\)/);
  assert.match(branch, /return vec4<f32>\(encodeSrgbTransfer\(falseColour\), 1\.0\);/);
  assert.doesNotMatch(branch, /acesFitted|extendedHdrShoulder|hubbleColour|exposure/);

  const glsl = postFragmentGLSL.slice(postFragmentGLSL.indexOf("void main()"));
  const glslBranch = glsl.indexOf("if (uDiagnosticDisplay > 0.5) {");
  assert.ok(glslBranch > 0 && glslBranch < glsl.indexOf("* exposure"));
  assert.match(postFragmentGLSL, /uniform float uDiagnosticDisplay;/);

  // The tracer decodes its sRGB palettes, so decode + encode is the identity.
  for (const value of [0, 0.02, 0.04045, 0.267, 0.5, 0.873, 1]) {
    assert.ok(Math.abs(encodeSrgb(decodeSrgb(value)) - value) < 1e-6);
  }
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /if \(mode >= 1 && mode <= 5\) \{\n    return decodeSrgbDisplay\(diagnosticColour\(result, mode\)\);/,
  );
  assert.match(
    strongFieldBinaryTraceFragmentWGSL,
    /return viridis\(result\.iterations \/ \(4\.0 \* f32\(stepBudget\) \+ 1\.0\)\);/,
  );
});

test("WebGPU uniform slot 26 carries the diagnostic display flag", () => {
  const written = [];
  const renderer = {
    uniformData: new Float32Array(40),
    width: 640,
    height: 360,
    outputHDR: false,
    displayP3: true,
    hdrPeak: 1,
    skyRadianceScale: 1,
    shaderBundle: {},
    device: { queue: { writeBuffer: (...args) => written.push(args) } },
  };
  WebGPURenderer.prototype.writeUniforms.call(renderer, frame());
  assert.equal(renderer.uniformData[26], 0, "motion must not select diagnostics");
  WebGPURenderer.prototype.writeUniforms.call(
    renderer,
    frame({ diagnosticDisplay: true }),
  );
  assert.equal(renderer.uniformData[26], 1);
  assert.equal(written.length, 2);
});

test("WebGL fallbacks without diagnostic modes keep the tone-mapped path", () => {
  const vector = () => ({ fromArray() {} });
  const traceUniforms = new Proxy({}, {
    get(target, name) {
      if (!(name in target)) {
        target[name] = { value: /Pos|Forward|Right|Up|Velocity/.test(name) ? vector() : 0 };
      }
      return target[name];
    },
  });
  const renderer = {
    traceUniforms,
    postUniforms: { uDiagnosticDisplay: { value: 0 } },
    shaderBundle: {},
  };
  WebGLRenderer.prototype.writeUniforms.call(renderer, frame({ diagnosticDisplay: true }));
  assert.equal(renderer.postUniforms.uDiagnosticDisplay.value, 1);
  for (const bundle of [strongFieldBinaryShaderBundle, strongFieldBinaryDualDiskShaderBundle]) {
    assert.equal(bundle.glsl.diagnosticModes, false);
    renderer.shaderBundle = { glsl: bundle.glsl };
    WebGLRenderer.prototype.writeUniforms.call(renderer, frame({ diagnosticDisplay: true }));
    assert.equal(renderer.postUniforms.uDiagnosticDisplay.value, 0);
  }
});

test("the HDR shoulder follows the current screen, not the startup screen", () => {
  const renderer = {
    uniformData: new Float32Array(40),
    width: 640,
    height: 360,
    outputHDR: true,
    displayP3: true,
    hdrPeak: 4,
    skyRadianceScale: 1,
    shaderBundle: {},
    device: { queue: { writeBuffer() {} } },
    dynamicRangeQuery: { matches: true },
    screenIsHDR: WebGPURenderer.prototype.screenIsHDR,
  };
  WebGPURenderer.prototype.writeUniforms.call(renderer, frame({ mode: 0 }));
  assert.equal(renderer.uniformData[32], 1);
  // Moving the window to an SDR screen switches to the tone-mapped path.
  renderer.dynamicRangeQuery.matches = false;
  WebGPURenderer.prototype.writeUniforms.call(renderer, frame({ mode: 0 }));
  assert.equal(renderer.uniformData[32], 0);
  renderer.outputHDR = false;
  renderer.dynamicRangeQuery.matches = true;
  WebGPURenderer.prototype.writeUniforms.call(renderer, frame({ mode: 0 }));
  assert.equal(renderer.uniformData[32], 0, "an SDR canvas never uses the shoulder");
});
