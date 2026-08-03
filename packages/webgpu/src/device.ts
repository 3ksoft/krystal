export interface WebGpuDeviceOptions {
  powerPreference?: GPUPowerPreference;
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
  label?: string;
}

export interface WebGpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
}

/**
 * Thin WebGPU bootstrap used by Chomato hosts.
 *
 * This deliberately owns no resource graph, shader DSL, schema integration,
 * or model behavior. Those belong to higher layers.
 */
export async function createWebGpuDevice(options: WebGpuDeviceOptions = {}): Promise<WebGpuContext> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable");

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference,
  });
  if (!adapter) throw new Error("Could not acquire a WebGPU adapter");

  const device = await adapter.requestDevice({
    label: options.label ?? "chomato",
    requiredFeatures: options.requiredFeatures ?? [],
    requiredLimits: options.requiredLimits ?? {},
  });

  return { adapter, device };
}
