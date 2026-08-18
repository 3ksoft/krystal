/**
 * Install the Dawn bindings once per process.
 *
 * `create([])` spins up a fresh Dawn instance. Doing that more than once in a
 * process — which happens as soon as two test files each bootstrap their own
 * `navigator.gpu` — aborts the native side with
 * `std::system_error: Invalid argument`. Every harness therefore shares the
 * first instance, and files that use `loadModel()` inherit whatever is already
 * installed.
 */
let installed: Promise<void> | undefined;

export function installDawn(): Promise<void> {
  installed ??= (async () => {
    const existing = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
    if (existing?.gpu) return;
    const { create, globals } = await import("webgpu");
    Object.assign(globalThis, globals);
    Object.defineProperty(globalThis, "navigator", {
      value: { gpu: create([]) },
      configurable: true,
    });
  })();
  return installed;
}
