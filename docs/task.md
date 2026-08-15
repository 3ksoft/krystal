Oto naturalne, techniczne tłumaczenie na język angielski (idealne np. do promptu dla LLM lub instrukcji dla programisty):

***

Your task is to implement the forward and backward passes in WebGPU for the custom Krystal model.

### Relevant local projects (we use both in this project):
* `schema-pop` – `/home/kr/Projects/schema-pop` (binary-exact schemas for WGSL)
* `sandblaster` – `/home/kr/Projects/sandblaster-v2` (typed WebGPU wrapper)

**Important!** There might be bugs in either project. If you encounter one, **report it to me before doing anything else!**

### Repository Context & Guidelines:
The `krystal` repo is a clone/fork of the WebGPU harness for the `lfm2.5` model. I haven't deleted any files, so everything should work as-is (models shouldn't be necessary, but if needed, they are at `/home/kr/Projects/debil-chomato/models`).

In `/home/kr/Projects/krystal/`, you can modify, overwrite, and delete files as needed. The final step of the task will be removing all redundant files. Everything is tracked in Git, so if you are sure that a particular file/directory inside `/home/kr/Projects/krystal/` is not needed for our architecture, feel free to delete it immediately to save mechanical cleanup work later.

### First Step (Infrastructure Setup):
First, ensure that the entire infrastructure is working properly: you must be able to run WebGPU test scripts from the console/CLI. Treat establishing this execution pipeline as your first task. Deno is likely a dead end here; prefer Bun/Node. You will definitely be using `flake.nix`—do not start the rest of the task without it.

### Python / Verification:
A decision was made to bypass Python, as we would have to write both forward and backward passes anyway. If you need Python to verify anything, look into `finetune` (Python + CUDA on a GeForce RTX 3060 12GB VRAM is available).

### Documentation & Reference:
How you organize your work from there is up to you. 

* **Main engine specification:** `docs/KRYSTAL_BRAIN_ARCHITECTURE_V2.md`
* **Pre-prepared schema:** `packages/schema/src/krystal-engine-schema.ts`
* **Initial Backward plan/description:** `docs/WEBGPU_BACKWARD_PLAN.md`