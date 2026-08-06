// Milestone 1: prove the scriptc toolchain works in this repo.
export function arenaElements(hidden: number, ff: number, vocab: number): number {
  // Mirror the shape of lfm2-definition.ts arena math.
  const work = (tokens: number) =>
    3 * tokens * hidden + 2 * tokens * Math.max(ff, 3 * hidden);
  const context = 1024;
  const repair = 8;
  return work(context) + work(repair) + vocab;
}

console.log("scriptc milestone 1 OK");
console.log(`arena elements = ${arenaElements(2048, 8192, 65536)}`);
