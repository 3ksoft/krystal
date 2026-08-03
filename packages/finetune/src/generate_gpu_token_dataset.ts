// generate_gpu_token_dataset.ts
//
// deno run --allow-write generate_gpu_token_dataset.ts
//
// Output:
//   ./gpu-token-dataset/train.jsonl
//   ./gpu-token-dataset/validation.jsonl
//
// Cel eksperymentu:
// nauczyć LFM rozróżniać:
//
//   "opowiedz mi o symulacji"      -> zwykła odpowiedź
//   "zasymuluj to"                -> <|reserved_100|>...<|reserved_101|>
//
// Nie próbujemy jeszcze uczyć prawdziwego GPU ISA.

const GPU_START = "<|reserved_100|>";
const GPU_END = "<|reserved_101|>";

const TRAIN_SIZE = 1000;
const VALIDATION_SIZE = 200;

const GPU_RATIO = 0.5;

const OUTPUT_DIR = "./packages/finetune/gpu-token-dataset";

type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

type Example = {
  messages: Message[];
  metadata: {
    kind: "gpu" | "normal";
    category: string;
  };
};

// ---------------------------------------------------------------------------
// Deterministyczny PRNG
// ---------------------------------------------------------------------------

class Random {
  constructor(private state = 0xC0FFEE) {}

  next(): number {
    // mulberry32
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(0, items.length - 1)]!;
  }

  chance(p: number): boolean {
    return this.next() < p;
  }

  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      [items[i], items[j]] = [items[j]!, items[i]!];
    }
    return items;
  }
}

const rng = new Random();

// ---------------------------------------------------------------------------
// System prompt
//
// Na początku zostawiłbym bardzo mało instrukcji.
// Chcemy sprawdzić, czy semantyka trafia do adaptera, a nie czy model nauczy się
// kopiować instrukcję z kilometrowego system prompta.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPTS = [
  "You are a helpful assistant.",
  "Answer the user's request.",
  "Respond appropriately to the user's request.",
] as const;

// ---------------------------------------------------------------------------
// GPU POSITIVES
// ---------------------------------------------------------------------------

const simulationObjects = [
  "particles",
  "rigid bodies",
  "water particles",
  "a cloth mesh",
  "projectiles",
  "aircraft",
  "cars",
  "planets",
  "agents",
  "fluid particles",
  "smoke particles",
  "balls",
] as const;

const gpuActions = [
  "simulate",
  "run a simulation of",
  "calculate the evolution of",
  "run",
  "evaluate",
] as const;

const resultRequests = [
  "and return the final state",
  "and tell me whether they collide",
  "and return the minimum distance",
  "and find the best result",
  "and report the final position",
  "and calculate the outcome",
  "",
] as const;

function makeGpuExample(): Example {
  const object = rng.pick(simulationObjects);
  const count = rng.pick([
    64,
    128,
    256,
    512,
    1000,
    1024,
    2048,
    4096,
    8192,
  ]);

  const steps = rng.pick([4, 8, 10, 16, 20, 32, 50, 64, 100, 128, 300]);
  const action = rng.pick(gpuActions);
  const result = rng.pick(resultRequests);

  const style = rng.int(0, 7);

  let user: string;

  switch (style) {
    case 0:
      user = `${capitalize(action)} ${count} ${object} for ${steps} steps ${result}.`;
      break;

    case 1:
      user =
        `I have ${count} ${object}. Run ${steps} simulation steps ${result}.`;
      break;

    case 2:
      user =
        `Can you actually simulate ${count} ${object} over ${steps} iterations ${result}?`;
      break;

    case 3:
      user =
        `Use the simulator to process ${count} ${object} for ${steps} steps ${result}.`;
      break;

    case 4:
      user =
        `Please compute what happens to ${count} ${object} after ${steps} simulation steps ${result}.`;
      break;

    case 5:
      user =
        `Run a numerical simulation: ${count} ${object}, ${steps} steps ${result}.`;
      break;

    case 6:
      user =
        `Don't explain how to simulate it. Actually run ${count} ${object} for ${steps} steps ${result}.`;
      break;

    default:
      user =
        `Evaluate the future state of ${count} ${object} by simulating ${steps} steps ${result}.`;
      break;
  }

  // Payload na tym etapie jest CELOWO banalny.
  //
  // Model ma nauczyć się:
  //
  // GPU_START -> payload -> GPU_END
  //
  // a nie jeszcze pełnej gramatyki.
  const payload =
    `simulate count=${count} object=${normalizeObject(object)} steps=${steps}`;

  return {
    messages: [
      {
        role: "system",
        content: rng.pick(SYSTEM_PROMPTS),
      },
      {
        role: "user",
        content: clean(user),
      },
      {
        role: "assistant",
        content: `${GPU_START}${payload}${GPU_END}`,
      },
    ],
    metadata: {
      kind: "gpu",
      category: "execute_simulation",
    },
  };
}

// ---------------------------------------------------------------------------
// HARD NEGATIVES
//
// To jest moim zdaniem najważniejsza część datasetu.
//
// Zawierają słowa:
// simulation / GPU / particles / WebGPU
//
// ale użytkownik NIE chce wykonania symulacji.
// ---------------------------------------------------------------------------

const hardNegativeTemplates = [
  {
    category: "explain_simulation",
    user: () =>
      `Explain how a simulation of ${rng.pick(simulationObjects)} works.`,
    assistant: () =>
      "A simulation numerically updates a model of the system over a sequence of time steps according to its rules and state.",
  },
  {
    category: "simulation_definition",
    user: () => "What is a particle simulation?",
    assistant: () =>
      "A particle simulation represents a system as many discrete particles whose properties, such as position and velocity, are updated over time.",
  },
  {
    category: "write_code",
    user: () =>
      `Write TypeScript code that simulates ${rng.pick(simulationObjects)}.`,
    assistant: () =>
      "You can represent the state in arrays and update positions and velocities in a loop using a fixed time step.",
  },
  {
    category: "write_shader",
    user: () =>
      `Write a WebGPU compute shader for simulating ${rng.pick(simulationObjects)}.`,
    assistant: () =>
      "A compute shader can assign one invocation to each element, read its current state from a storage buffer, update it, and write the new state back.",
  },
  {
    category: "gpu_question",
    user: () => "Why are GPUs useful for simulations?",
    assistant: () =>
      "GPUs are well suited to simulations with large amounts of parallel work because many independent elements can be updated concurrently.",
  },
  {
    category: "estimate",
    user: () =>
      `Roughly how expensive would a ${rng.pick([1000, 10000, 100000])}-particle simulation be?`,
    assistant: () =>
      "The cost depends heavily on the algorithm, number of interactions, memory layout, number of steps, and target hardware.",
  },
  {
    category: "hypothetical",
    user: () =>
      `What would probably happen if ${rng.pick(simulationObjects)} were simulated for a very long time?`,
    assistant: () =>
      "The outcome depends on the forces, constraints, boundary conditions, numerical integration method, and stability of the simulation.",
  },
  {
    category: "describe_algorithm",
    user: () =>
      `Describe an algorithm for simulating ${rng.pick(simulationObjects)}.`,
    assistant: () =>
      "A typical algorithm initializes the state, computes forces or constraints, integrates the state forward, resolves interactions, and repeats for each time step.",
  },
  {
    category: "compare",
    user: () => "Compare CPU and GPU particle simulation.",
    assistant: () =>
      "CPU implementations are often easier to control and debug, while GPU implementations can provide much higher throughput for highly parallel workloads.",
  },
  {
    category: "debug",
    user: () =>
      "My particle simulation explodes after a few frames. What could cause that?",
    assistant: () =>
      "Common causes include an excessively large time step, unstable integration, incorrect constraint parameters, invalid values, or forces that grow without bounds.",
  },
] as const;

function makeHardNegative(): Example {
  const item = rng.pick(hardNegativeTemplates);

  return {
    messages: [
      {
        role: "system",
        content: rng.pick(SYSTEM_PROMPTS),
      },
      {
        role: "user",
        content: item.user(),
      },
      {
        role: "assistant",
        content: item.assistant(),
      },
    ],
    metadata: {
      kind: "normal",
      category: item.category,
    },
  };
}

// ---------------------------------------------------------------------------
// EASY NORMAL EXAMPLES
//
// Potrzebne, żeby LoRA nie dostała urojenia:
// "wszystko -> GPU token"
// ---------------------------------------------------------------------------

const easyNormalExamples = [
  [
    "What is 2 + 2?",
    "4",
  ],
  [
    "Say hello in Polish.",
    "Cześć!",
  ],
  [
    "What is the capital of France?",
    "Paris.",
  ],
  [
    "Explain recursion in one sentence.",
    "Recursion is a technique where a function solves a problem by calling itself on smaller instances of the same problem.",
  ],
  [
    "Write a short greeting.",
    "Hello! Nice to meet you.",
  ],
  [
    "What does HTTP stand for?",
    "HTTP stands for Hypertext Transfer Protocol.",
  ],
  [
    "Is 17 a prime number?",
    "Yes, 17 is a prime number.",
  ],
  [
    "Give me three colors.",
    "Red, green, and blue.",
  ],
  [
    "What is a GPU?",
    "A GPU is a processor designed to execute large amounts of parallel numerical work efficiently.",
  ],
  [
    "What is WebGPU?",
    "WebGPU is a modern graphics and compute API that provides applications with access to GPU capabilities.",
  ],
  [
    "Explain what a compute shader is.",
    "A compute shader is a GPU program designed for general parallel computation rather than directly rendering graphics.",
  ],
] as const;

function makeEasyNormal(): Example {
  const [user, assistant] = rng.pick(easyNormalExamples);

  return {
    messages: [
      {
        role: "system",
        content: rng.pick(SYSTEM_PROMPTS),
      },
      {
        role: "user",
        content: user,
      },
      {
        role: "assistant",
        content: assistant,
      },
    ],
    metadata: {
      kind: "normal",
      category: "ordinary_response",
    },
  };
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

function generateDataset(size: number): Example[] {
  const gpuCount = Math.round(size * GPU_RATIO);
  const normalCount = size - gpuCount;

  const examples: Example[] = [];

  for (let i = 0; i < gpuCount; i++) {
    examples.push(makeGpuExample());
  }

  // Większość negatywów robimy "hard".
  // Chcemy koniecznie rozdzielić:
  //
  //   "symulacja" != "wykonaj symulację"
  //
  const hardCount = Math.round(normalCount * 0.7);
  const easyCount = normalCount - hardCount;

  for (let i = 0; i < hardCount; i++) {
    examples.push(makeHardNegative());
  }

  for (let i = 0; i < easyCount; i++) {
    examples.push(makeEasyNormal());
  }

  return rng.shuffle(examples);
}

// ---------------------------------------------------------------------------
// Validation
//
// Osobny generator i inny seed nie są tu niezbędne, ponieważ kombinatoryka
// promptów jest spora, ale dodatkowo pilnujemy braku identycznych user promptów.
// ---------------------------------------------------------------------------

function removeUserPromptDuplicates(
  train: Example[],
  validation: Example[],
): Example[] {
  const trainPrompts = new Set(
    train.map((x) =>
      x.messages.find((message) => message.role === "user")!.content
    ),
  );

  return validation.filter((x) => {
    const prompt = x.messages.find((message) => message.role === "user")!
      .content;

    return !trainPrompts.has(prompt);
  });
}

// ---------------------------------------------------------------------------

function clean(value: string): string {
  return value
    .replace(/\s+/g, " ")
    .replace(/\s+\./g, ".")
    .trim();
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function normalizeObject(value: string): string {
  return value.replace(/\s+/g, "_");
}

async function writeJsonl(path: string, examples: Example[]) {
  const text = examples
    .map((example) => JSON.stringify(example))
    .join("\n") + "\n";

  await Deno.writeTextFile(path, text);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

await Deno.mkdir(OUTPUT_DIR, { recursive: true });

const train = generateDataset(TRAIN_SIZE);

let validation = generateDataset(
  // trochę więcej, bo odfiltrujemy przypadkowe duplikaty
  VALIDATION_SIZE + 100,
);

validation = removeUserPromptDuplicates(train, validation).slice(
  0,
  VALIDATION_SIZE,
);

await writeJsonl(`${OUTPUT_DIR}/train.jsonl`, train);
await writeJsonl(`${OUTPUT_DIR}/validation.jsonl`, validation);

const trainGpu = train.filter((x) => x.metadata.kind === "gpu").length;
const trainNormal = train.length - trainGpu;

const valGpu = validation.filter((x) => x.metadata.kind === "gpu").length;
const valNormal = validation.length - valGpu;

console.log("======================================================");
console.log(" GPU TOKEN DATASET");
console.log("======================================================");
console.log(`GPU_START: ${GPU_START}`);
console.log(`GPU_END:   ${GPU_END}`);
console.log("");
console.log(`train:      ${train.length}`);
console.log(`  gpu:      ${trainGpu}`);
console.log(`  normal:   ${trainNormal}`);
console.log("");
console.log(`validation: ${validation.length}`);
console.log(`  gpu:      ${valGpu}`);
console.log(`  normal:   ${valNormal}`);
console.log("");
console.log(`${OUTPUT_DIR}/train.jsonl`);
console.log(`${OUTPUT_DIR}/validation.jsonl`);
console.log("======================================================");