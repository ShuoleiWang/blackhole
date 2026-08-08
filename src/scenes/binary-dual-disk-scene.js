import { createBinaryScene } from "./binary-approx-scene.js";

/**
 * Strong-field binary lensing with two analytic, tidally truncated thin
 * mini-disks. The shared binary factory keeps the vacuum route as the stable
 * control while this wrapper selects the separate emission product layer.
 */
export function createBinaryDualDiskScene(options) {
  return createBinaryScene(options, "dual-disk");
}
