// Reduced WebGL2 fallback shared by the transient scenes: the sky and up to
// two limb-darkened spheres (stars, or the photosphere) coloured on the CPU
// from the scene's physical model. It has no volumetric transport, lensing,
// Doppler or light-travel terms, and the scenes label it as such.

export const TRANSIENT_SPHERES_FALLBACK_GLSL = /* glsl */ `
precision highp float;
precision highp int;

varying vec2 vUv;
uniform vec2 uResolution;
uniform vec3 uCameraPos;
uniform vec3 uForward;
uniform float uFov;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uSkyRotation;
uniform float uSkyRadianceScale;
uniform sampler2D tSky;
uniform vec4 uSceneStarA;
uniform vec4 uSceneStarB;
uniform vec4 uSceneColourA;
uniform vec4 uSceneColourB;
uniform vec4 uSceneSky;

const float PI = 3.14159265358979323846;

vec2 sphereHit(vec3 origin, vec3 direction, vec4 sphere) {
  vec3 offset = origin - sphere.xyz;
  float b = dot(offset, direction);
  float c = dot(offset, offset) - sphere.w * sphere.w;
  float discriminant = b * b - c;
  if (discriminant <= 0.0 || sphere.w <= 0.0) {
    return vec2(1.0e30, 0.0);
  }
  float root = sqrt(discriminant);
  float entry = -b - root;
  return entry > 0.0 ? vec2(entry, 1.0) : vec2(1.0e30, 0.0);
}

vec3 skyColour(vec3 direction) {
  float c = cos(uSkyRotation);
  float s = sin(uSkyRotation);
  vec3 d = normalize(vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z));
  vec2 uv = vec2(fract(atan(d.z, d.x) / (2.0 * PI) + 0.5), clamp(0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI, 0.00001, 0.99999));
  return max(texture2D(tSky, uv).rgb - vec3(0.0015), vec3(0.0)) * uSkyRadianceScale * uSceneSky.x;
}

vec3 shade(vec3 origin, vec3 direction, vec4 sphere, vec4 colour, float distance) {
  vec3 normal = normalize(origin + distance * direction - sphere.xyz);
  float mu = clamp(dot(normal, -direction), 0.0, 1.0);
  return colour.rgb * (1.0 - colour.w * (1.0 - mu));
}

void main() {
  float aspect = uResolution.x / uResolution.y;
  vec2 screen = vec2((vUv.x * 2.0 - 1.0) * aspect, vUv.y * 2.0 - 1.0);
  float tanHalfFov = tan(0.5 * clamp(uFov, 0.02, 2.8));
  vec3 forward = normalize(uForward);
  vec3 right = normalize(uRight - forward * dot(forward, uRight));
  vec3 up = cross(right, forward);
  vec3 direction = normalize(forward + tanHalfFov * (screen.x * right + screen.y * up));
  vec2 hitA = sphereHit(uCameraPos, direction, uSceneStarA);
  vec2 hitB = sphereHit(uCameraPos, direction, uSceneStarB);
  vec3 colour = skyColour(direction);
  if (hitA.y > 0.5 && hitA.x <= hitB.x) {
    colour = shade(uCameraPos, direction, uSceneStarA, uSceneColourA, hitA.x);
  } else if (hitB.y > 0.5) {
    colour = shade(uCameraPos, direction, uSceneStarB, uSceneColourB, hitB.x);
  }
  gl_FragColor = vec4(min(colour, vec3(60000.0)), 1.0);
}
`;

// Scene uniforms of the fallback, filled from frame.sceneTransientFallback =
// { starA, starB, colourA, colourB, sky } (vec4 each: sphere centre and
// radius in scene units; exposed linear-sRGB radiance and limb-darkening
// coefficient; sky scale).
export function transientFallbackUniforms(sceneName) {
  return {
    createWebGLExtras(THREE) {
      return {
        uSceneStarA: { value: new THREE.Vector4() },
        uSceneStarB: { value: new THREE.Vector4() },
        uSceneColourA: { value: new THREE.Vector4() },
        uSceneColourB: { value: new THREE.Vector4() },
        uSceneSky: { value: new THREE.Vector4() },
      };
    },
    writeWebGLExtras(uniforms, frame) {
      const fallback = frame?.sceneTransientFallback;
      if (!fallback) {
        throw new Error(`${sceneName} WebGL2 fallback requires sceneTransientFallback`);
      }
      uniforms.uSceneStarA.value.fromArray(fallback.starA);
      uniforms.uSceneStarB.value.fromArray(fallback.starB);
      uniforms.uSceneColourA.value.fromArray(fallback.colourA);
      uniforms.uSceneColourB.value.fromArray(fallback.colourB);
      uniforms.uSceneSky.value.fromArray(fallback.sky);
    },
  };
}
