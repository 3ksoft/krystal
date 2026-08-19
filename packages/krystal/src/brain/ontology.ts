export class OntologyGraph {
  private readonly parentToChildren = new Map<number, Set<number>>();
  private readonly childToParents = new Map<number, Set<number>>();
  private readonly symbolToToken = new Map<string, number>();

  constructor(tokenBySymbol: ReadonlyMap<string, number>) {
    for (const [sym, token] of tokenBySymbol.entries()) {
      this.symbolToToken.set(sym, token);
    }
  }

  public registerHierarchy(parentToken: number, childToken: number): void {
    if (!this.parentToChildren.has(parentToken)) {
      this.parentToChildren.set(parentToken, new Set());
    }
    this.parentToChildren.get(parentToken)!.add(childToken);

    if (!this.childToParents.has(childToken)) {
      this.childToParents.set(childToken, new Set());
    }
    this.childToParents.get(childToken)!.add(parentToken);
  }

  public isA(derivedToken: number, baseCategoryToken: number): boolean {
    if (derivedToken === baseCategoryToken) return true;

    const visited = new Set<number>();
    const queue: number[] = [derivedToken];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (current === baseCategoryToken) return true;
      if (visited.has(current)) continue;
      visited.add(current);

      const parents = this.childToParents.get(current);
      if (parents) {
        for (const parent of parents) {
          if (!visited.has(parent)) queue.push(parent);
        }
      }
    }

    return false;
  }

  public getToken(symbol: string): number | undefined {
    return this.symbolToToken.get(symbol);
  }
}
