export interface WebGpuDeviceOptions {
  powerPreference?: GPUPowerPreference;
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
  label?: string;
  /** Request `timestamp-query` when the adapter supports it, for GPU profiling. */
  timestampQuery?: boolean;
}

export interface WebGpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
}

/**
 * Thin WebGPU bootstrap used by Krystal hosts.
 *
 * This deliberately owns no resource graph, shader DSL, schema integration,
 * or model behavior. Those belong to higher layers.
 */
export async function createWebGpuDevice(options: WebGpuDeviceOptions = {}): Promise<WebGpuContext> {
  // Reached through globalThis rather than the bare `navigator` binding: this
  // module is compiled for runtimes that have no such global (a scriptc-built
  // native exe among them), where naming it directly is a hard type error even
  // though the code path is never taken there.
  const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
  if (!gpu) throw new Error("WebGPU is unavailable");

  const adapter = await gpu.requestAdapter(
    options.powerPreference === undefined
      ? {}
      : { powerPreference: options.powerPreference },
  );
  if (!adapter) throw new Error("Could not acquire a WebGPU adapter");

  const features = [...(options.requiredFeatures ?? [])];
  // `timestamp-query` is optional and not universally available, so it is
  // requested only when asked for and only when the adapter has it. Enabling
  // the feature does not instrument anything by itself — it merely makes GPU
  // pass timing attachable later.
  if (options.timestampQuery && adapter.features.has("timestamp-query")) {
    if (!features.includes("timestamp-query")) features.push("timestamp-query");
  }

  const device = await adapter.requestDevice({
    label: options.label ?? "krystal",
    requiredFeatures: features,
    requiredLimits: options.requiredLimits ?? {},
  });

  return { adapter, device };
}
