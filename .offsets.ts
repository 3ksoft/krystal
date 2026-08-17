import { KRYSTAL_FORWARD_ARENA } from "./packages/webgpu/src/krystal-layout.ts";
const names = ["fieldStates","encQ","encK","encV","encOut","encH1","encMask","encP","bankKeys","bankValues","queryKeys","queryValues","mixerQ","mixerK","mixerV","mixerH1","mixed","mixerMask","mixerP","selectorQ","selectorK","intentMask","argMask","intentP","intentGather","intentIndices","argP","argGather","argIndices"];
let prev = 0;
for (const n of names) {
  const off = (KRYSTAL_FORWARD_ARENA as any)[n];
  console.log(`${n.padEnd(14)} offset=${off} size_diff=${off - prev}`);
  prev = off;
}
console.log("total elements:", (KRYSTAL_FORWARD_ARENA as any).elements);
