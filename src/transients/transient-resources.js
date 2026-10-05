// Static RGBA32F tables (blackbody colour, radiation-diffusion solutions,
// composition) shared by a transient scene's WebGPU tracer and its WebGL2
// fallback. Binding 3 is left for an optional sparse coarse field.

function validateTable(table, name) {
  if (
    !table
    || !Number.isInteger(table.width)
    || !Number.isInteger(table.height)
    || table.width < 1
    || table.height < 1
    || !(table.data instanceof Float32Array)
    || table.data.length !== table.width * table.height * 4
    || !table.data.every(Number.isFinite)
  ) {
    throw new Error(`Transient table ${name} must be a finite RGBA32F width x height array`);
  }
}

/**
 * @param {Array<{ name: string, binding: number, uniform: string, width: number, height: number, data: Float32Array }>} tables
 */
export function createTransientTableResources(tables) {
  const bindings = new Set();
  for (const table of tables) {
    validateTable(table, table.name);
    if (!Number.isInteger(table.binding) || table.binding < 4 || bindings.has(table.binding)) {
      throw new Error(`Transient table ${table.name} needs a unique binding >= 4`);
    }
    bindings.add(table.binding);
  }
  return Object.freeze({
    createWebGPU(device) {
      const textures = tables.map((table) => {
        const texture = device.createTexture({
          label: `Transient table · ${table.name}`,
          size: [table.width, table.height, 1],
          format: "rgba32float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
          { texture },
          table.data,
          { bytesPerRow: table.width * 16, rowsPerImage: table.height },
          [table.width, table.height, 1],
        );
        return texture;
      });
      return {
        entries: tables.map((table, index) => ({
          binding: table.binding,
          resource: textures[index].createView(),
        })),
        dispose() {
          for (const texture of textures) {
            texture.destroy();
          }
        },
      };
    },
    createWebGL(THREE) {
      const textures = tables.map((table) => {
        const texture = new THREE.DataTexture(
          table.data,
          table.width,
          table.height,
          THREE.RGBAFormat,
          THREE.FloatType,
        );
        texture.minFilter = THREE.NearestFilter;
        texture.magFilter = THREE.NearestFilter;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.generateMipmaps = false;
        texture.needsUpdate = true;
        return texture;
      });
      return {
        uniforms: Object.fromEntries(tables.map((table, index) => [
          table.uniform,
          { value: textures[index] },
        ])),
        dispose() {
          for (const texture of textures) {
            texture.dispose();
          }
        },
      };
    },
  });
}
