import type {
  JsonNode,
  LayoutConstraintProgram,
  LayoutConstraintProgramSummary,
} from "./index.ts";

const encoder = new TextEncoder();

/**
 * Limits for the JSON-Schema frontend. These are language/ABI limits rather
 * than memory-layout limits; bounded dynamic arrays are intentionally allowed.
 */
export const JSON_SCHEMA_CONSTRAINT_LIMITS = {
  maxArrayItems: 0xff,
  maxEnumVariants: 0xff,
  maxStringLength: 0xffff,
  maxNumberChars: 64,
} as const;

export class UnsupportedJsonSchemaConstraintError extends Error {
  readonly path: string;

  constructor(path: string, reason: string) {
    super(`[json-schema-constraint] ${path}: ${reason}`);
    this.path = path;
    this.name = "UnsupportedJsonSchemaConstraintError";
  }
}

type JsonSchemaObject = Record<string, unknown>;

function asSchema(value: unknown, path: string): JsonSchemaObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new UnsupportedJsonSchemaConstraintError(path, "expected a schema object");
  }
  return value as JsonSchemaObject;
}

function finiteNonNegativeInteger(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value) || value < 0) return undefined;
  return value;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function schemaType(schema: JsonSchemaObject, path: string): string | undefined {
  const value = schema.type;
  if (value === undefined) return undefined;
  if (typeof value === "string") return value;
  throw new UnsupportedJsonSchemaConstraintError(path, "union-valued 'type' is not supported yet");
}

function encodeJsonLiteral(value: unknown, path: string): string {
  if (typeof value === "string" || typeof value === "boolean" || value === null) return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  throw new UnsupportedJsonSchemaConstraintError(path, "enum/const supports only finite JSON scalar values");
}

/**
 * Compile bounded JSON Schema directly into the same byte-level constraint IR
 * used by the LayoutPlan frontend.
 *
 * This frontend exists because binary LayoutPlan is intentionally a fixed
 * memory-layout IR and may erase bounded-but-dynamic JSON constructs such as
 * `maxItems`. Structured generation must preserve those language constraints.
 *
 * Arrays are unrolled by count, not by enumerating every possible complete
 * array. The graph therefore grows O(maxItems * itemProgram), while `split`
 * nodes choose between closing the array and emitting one more item.
 */
export function compileJsonSchemaProgram(rawSchema: unknown): LayoutConstraintProgram {
  const root = asSchema(rawSchema, "$");
  const nodes: JsonNode[] = [];
  const summary: LayoutConstraintProgramSummary = {
    rootType: "$",
    segments: 0,
    fields: 0,
    optionalIncluded: 0,
    optionalSkipped: 0,
    enums: 0,
    strings: 0,
    numbers: 0,
    booleans: 0,
    arrays: 0,
  };
  const counted = new Set<string>();

  const countOnce = (key: string, fn: () => void): void => {
    if (counted.has(key)) return;
    counted.add(key);
    fn();
  };

  const addNode = (node: JsonNode): number => {
    const id = nodes.length;
    nodes.push(node);
    return id;
  };

  const literal = (text: string, label: string, next: number): number =>
    addNode({ kind: "literal", bytes: encoder.encode(text), text, label, next });

  const choice = (texts: readonly string[], label: string, next: number): number => {
    const unique = [...new Set(texts)];
    if (unique.length === 0) throw new UnsupportedJsonSchemaConstraintError(label, "empty choice");
    if (unique.length > JSON_SCHEMA_CONSTRAINT_LIMITS.maxEnumVariants) {
      throw new UnsupportedJsonSchemaConstraintError(
        label,
        `choice has ${unique.length} variants; max is ${JSON_SCHEMA_CONSTRAINT_LIMITS.maxEnumVariants}`,
      );
    }
    return addNode({
      kind: "choice",
      alternatives: unique.map((value) => encoder.encode(value)),
      texts: unique,
      label,
      next,
    });
  };

  const resolveRef = (ref: string, path: string): JsonSchemaObject => {
    if (!ref.startsWith("#/")) {
      throw new UnsupportedJsonSchemaConstraintError(path, `only local JSON Pointer $ref is supported, got ${JSON.stringify(ref)}`);
    }
    let cursor: unknown = root;
    for (const rawPart of ref.slice(2).split("/")) {
      const part = rawPart.replace(/~1/g, "/").replace(/~0/g, "~");
      if (!cursor || typeof cursor !== "object" || Array.isArray(cursor) || !(part in cursor)) {
        throw new UnsupportedJsonSchemaConstraintError(path, `unresolved $ref ${JSON.stringify(ref)}`);
      }
      cursor = (cursor as JsonSchemaObject)[part];
    }
    return asSchema(cursor, `${path}->$ref(${ref})`);
  };

  let compileSchema: (
    schema: JsonSchemaObject,
    path: string,
    next: number,
    refStack: ReadonlySet<string>,
  ) => number;

  const compileObject = (
    schema: JsonSchemaObject,
    path: string,
    next: number,
    refStack: ReadonlySet<string>,
  ): number => {
    const propertiesRaw = schema.properties ?? {};
    const properties = asSchema(propertiesRaw, `${path}.properties`);
    const fields = Object.entries(properties);
    const required = new Set(
      Array.isArray(schema.required)
        ? schema.required.map((name) => {
            if (typeof name !== "string") {
              throw new UnsupportedJsonSchemaConstraintError(path, "required must contain property names");
            }
            return name;
          })
        : [],
    );

    for (const name of required) {
      if (!(name in properties)) {
        throw new UnsupportedJsonSchemaConstraintError(path, `required property ${JSON.stringify(name)} is missing from properties`);
      }
    }

    const close = literal("}", `object-close ${path}`, next);
    const memo = new Map<string, number>();

    const compileFields = (index: number, emitted: boolean): number => {
      if (index >= fields.length) return close;
      const memoKey = `${index}:${emitted ? 1 : 0}`;
      const cached = memo.get(memoKey);
      if (cached !== undefined) return cached;

      const [name, childRaw] = fields[index]!;
      const childPath = `${path}.${name}`;
      const child = asSchema(childRaw, childPath);
      const optional = !required.has(name);
      const afterPresent = compileFields(index + 1, true);
      const valueEntry = compileSchema(child, childPath, afterPresent, refStack);
      const propertyPrefix = `${emitted ? "," : ""}${JSON.stringify(name)}:`;
      const presentEntry = literal(propertyPrefix, `field ${childPath}`, valueEntry);

      countOnce(`field:${childPath}`, () => {
        summary.fields++;
        if (optional) summary.optionalIncluded++;
      });

      let entry = presentEntry;
      if (optional) {
        const skippedEntry = compileFields(index + 1, emitted);
        entry = addNode({
          kind: "split",
          targets: [presentEntry, skippedEntry],
          label: `optional ${childPath}`,
        });
      }
      memo.set(memoKey, entry);
      return entry;
    };

    return literal("{", `object-open ${path}`, compileFields(0, false));
  };

  const compileArray = (
    schema: JsonSchemaObject,
    path: string,
    next: number,
    refStack: ReadonlySet<string>,
  ): number => {
    if (schema.prefixItems !== undefined) {
      throw new UnsupportedJsonSchemaConstraintError(path, "prefixItems/tuple arrays are not supported yet");
    }
    const item = asSchema(schema.items, `${path}.items`);
    const minItems = finiteNonNegativeInteger(schema.minItems) ?? 0;
    const maxItems = finiteNonNegativeInteger(schema.maxItems);
    if (maxItems === undefined) {
      throw new UnsupportedJsonSchemaConstraintError(path, "array requires a finite integer maxItems");
    }
    if (maxItems > JSON_SCHEMA_CONSTRAINT_LIMITS.maxArrayItems) {
      throw new UnsupportedJsonSchemaConstraintError(
        path,
        `maxItems=${maxItems}; max is ${JSON_SCHEMA_CONSTRAINT_LIMITS.maxArrayItems}`,
      );
    }
    if (minItems > maxItems) {
      throw new UnsupportedJsonSchemaConstraintError(path, `minItems=${minItems} exceeds maxItems=${maxItems}`);
    }

    countOnce(`array:${path}`, () => summary.arrays++);
    const close = literal("]", `array-close ${path}`, next);
    const memo = new Map<number, number>();

    const compileCount = (count: number): number => {
      const cached = memo.get(count);
      if (cached !== undefined) return cached;
      if (count === maxItems) return close;

      const afterItem = compileCount(count + 1);
      let itemEntry = compileSchema(item, `${path}[${count}]`, afterItem, refStack);
      if (count > 0) itemEntry = literal(",", `array-comma ${path}[${count}]`, itemEntry);

      const decision = count >= minItems
        ? addNode({
            kind: "split",
            targets: [close, itemEntry],
            label: `array-count ${path} ${count}/${minItems}..${maxItems}`,
          })
        : itemEntry;

      // Keep each bounded count as an explicit program-counter state. Without
      // this epsilon barrier the GPU linker can statically flatten an empty or
      // fully-optional item into the following count, creating a combinatorial
      // trie across many array positions.
      const entry = addNode({
        kind: "jump",
        next: decision,
        label: `array-barrier ${path} ${count}`,
      });
      memo.set(count, entry);
      return entry;
    };

    return literal("[", `array-open ${path}`, compileCount(0));
  };

  compileSchema = (
    schema: JsonSchemaObject,
    path: string,
    next: number,
    refStack: ReadonlySet<string>,
  ): number => {
    if (typeof schema.$ref === "string") {
      if (refStack.has(schema.$ref)) {
        throw new UnsupportedJsonSchemaConstraintError(path, `cyclic $ref ${JSON.stringify(schema.$ref)}`);
      }
      const nextStack = new Set(refStack);
      nextStack.add(schema.$ref);
      return compileSchema(resolveRef(schema.$ref, path), path, next, nextStack);
    }

    if (schema.const !== undefined) {
      return literal(encodeJsonLiteral(schema.const, path), `const ${path}`, next);
    }

    if (Array.isArray(schema.enum)) {
      countOnce(`enum:${path}`, () => summary.enums++);
      return choice(schema.enum.map((value) => encodeJsonLiteral(value, path)), `enum ${path}`, next);
    }

    const type = schemaType(schema, path);
    switch (type) {
      case "object":
        return compileObject(schema, path, next, refStack);

      case "array":
        return compileArray(schema, path, next, refStack);

      case "string": {
        const minLength = finiteNonNegativeInteger(schema.minLength) ?? 0;
        const maxLength = finiteNonNegativeInteger(schema.maxLength);
        if (maxLength === undefined) {
          throw new UnsupportedJsonSchemaConstraintError(path, "string requires a finite integer maxLength");
        }
        if (maxLength > JSON_SCHEMA_CONSTRAINT_LIMITS.maxStringLength) {
          throw new UnsupportedJsonSchemaConstraintError(
            path,
            `maxLength=${maxLength}; max is ${JSON_SCHEMA_CONSTRAINT_LIMITS.maxStringLength}`,
          );
        }
        if (minLength > maxLength) {
          throw new UnsupportedJsonSchemaConstraintError(path, `minLength=${minLength} exceeds maxLength=${maxLength}`);
        }
        countOnce(`string:${path}`, () => summary.strings++);
        return addNode({ kind: "string", minLength, maxLength, label: `string ${path}`, next });
      }

      case "integer":
      case "number": {
        const minimum = finiteNumber(schema.minimum);
        const maximum = finiteNumber(schema.maximum);
        const exclusiveMinimum = finiteNumber(schema.exclusiveMinimum);
        const exclusiveMaximum = finiteNumber(schema.exclusiveMaximum);
        if (exclusiveMinimum !== undefined || exclusiveMaximum !== undefined) {
          throw new UnsupportedJsonSchemaConstraintError(path, "exclusiveMinimum/exclusiveMaximum are not supported yet");
        }
        const step = finiteNumber(schema.multipleOf);
        if (schema.minimum !== undefined && minimum === undefined) {
          throw new UnsupportedJsonSchemaConstraintError(path, "minimum must be finite");
        }
        if (schema.maximum !== undefined && maximum === undefined) {
          throw new UnsupportedJsonSchemaConstraintError(path, "maximum must be finite");
        }
        if (step !== undefined && step <= 0) {
          throw new UnsupportedJsonSchemaConstraintError(path, "multipleOf must be > 0");
        }
        countOnce(`number:${path}`, () => summary.numbers++);
        return addNode({
          kind: "number",
          integer: type === "integer",
          min: minimum,
          max: maximum,
          step,
          maxChars: JSON_SCHEMA_CONSTRAINT_LIMITS.maxNumberChars,
          label: `number ${path}`,
          next,
        });
      }

      case "boolean":
        countOnce(`bool:${path}`, () => summary.booleans++);
        return choice(["true", "false"], `bool ${path}`, next);

      case "null":
        return literal("null", `null ${path}`, next);

      case undefined:
        throw new UnsupportedJsonSchemaConstraintError(path, "schema has neither type, enum, const nor $ref");

      default:
        throw new UnsupportedJsonSchemaConstraintError(path, `type ${JSON.stringify(type)} is not supported`);
    }
  };

  const accept = addNode({ kind: "accept", label: "<complete>" });
  const entry = compileSchema(root, "$", accept, new Set());
  summary.segments = nodes.length;
  return { nodes, entry, accept, summary };
}
