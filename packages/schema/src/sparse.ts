import { scope } from "arktype";
import { wgsl } from "@schema-pop/schema";


/**
 * Reserved bit patterns shared by the compiled schema and guide runtime.
 *
 * The LFM2 tokenizer currently uses IDs only through 64401, while the model
 * output space is 0..65535. 0xffff is therefore deliberately kept outside the
 * tokenizer and globally forbidden from sampling so fixed-size GPU tables can
 * use it as an empty slot without a parallel validity bitmap.
 *
 * Keep the names semantic even though the bit pattern is shared: a missing
 * node is not conceptually a token.
 */
export const GPU_SCHEMA_SENTINELS = {
	emptyToken: 0xffff,
	noneNode: 0xffff,
	noneIndex: 0xffff,
} as const;

export const GPU_SCHEMA_LIMITS = {
	/**
	 * Total number of schema nodes.
	 *
	 * u16 index space is intentionally used instead of u8 because nested
	 * schemas can exceed 255 nodes surprisingly easily.
	 *
	 * This is a count, not a maximum index: 0xffff nodes occupy indices
	 * 0x0000..0xfffe, leaving 0xffff available as NONE_NODE.
	 */
	maxNodes: 0xffff,

	/** Fields belonging to a single object/struct. */
	maxFieldsPerObject: 0xff,

	/** Variants belonging to a single enum. */
	maxEnumVariants: 0xff,

	/** Variants belonging to a single union. */
	maxUnionVariants: 0xff,

	/**
	 * Maximum number of tokenizer tokens representing one identifier/literal.
	 *
	 * Examples:
	 *   "temperature"
	 *   "sensor_reading"
	 *   "some-long-enum-value"
	 */
	maxTokensPerSpan: 0xff,

	/**
	 * Total token pool.
	 *
	 * Offset is u32, so this is primarily a build-time sanity limit rather
	 * than a representational limitation.
	 */
	maxTokenCount: 0xffffffff,
} as const;

export const $gpuSchema = scope({
	...wgsl.import(),

	// ---------------------------------------------------------------------
	// Primitive enums
	// ---------------------------------------------------------------------

	SchemaNodeKind:
		"'unit' | 'boolean' | 'integer' | 'number' | 'string' | 'array' | 'object' | 'enum' | 'union' | 'literal'",

	IntegerKind:
		"'i8' | 'u8' | 'i16' | 'u16' | 'i32' | 'u32' | 'i64' | 'u64'",

	NumberKind:
		"'f16' | 'f32' | 'f64'",

	LiteralKind:
		"'string' | 'integer' | 'number' | 'boolean' | 'null'",

	/**
	 * Union discrimination strategy.
	 *
	 * none:
	 *   generic union / alternatives
	 *
	 * tagged:
	 *   object variants differentiated by a discriminant field
	 */
	UnionKind: "'none' | 'tagged'",

	// ---------------------------------------------------------------------
	// Shared references
	// ---------------------------------------------------------------------

	/**
	 * Span inside SchemaTokens.
	 *
	 * tokenOffset points to logical u16 token IDs, not packed u32 words.
	 * tokenLength is therefore measured in tokenizer tokens.
	 *
	 * Storage may physically pack:
	 *
	 *   token[0] | token[1] -> one u32
	 *
	 * but all schema references use logical token indices.
	 */
	TokenSpan: {
		tokenOffset: "u32",
		tokenLength: "u8",

		/**
		 * Explicit padding keeps the struct WGSL-friendly and leaves room
		 * for future flags without changing its size.
		 */
		reserved0: "u8",
		reserved1: "u16",
	},

	/**
	 * Generic contiguous range.
	 *
	 * Used where the child table is obvious from the context.
	 */
	SchemaRange: {
		offset: "u32",
		count: "u16",
		reserved: "u16",
	},

	// ---------------------------------------------------------------------
	// Top-level schema metadata
	// ---------------------------------------------------------------------

	/**
	 * Header for one compiled schema.
	 *
	 * All offsets below are logical element offsets into the corresponding
	 * tables, not byte offsets.
	 */
	SchemaHeader: {
		/** Root SchemaNode index. */
		rootNode: "u16",

		/** Format version, currently 1. */
		version: "u16",

		nodeCount: "u16",
		objectFieldCount: "u16",

		enumVariantCount: "u16",
		unionVariantCount: "u16",

		/**
		 * Number of logical u16 token IDs in SchemaTokens.
		 */
		tokenCount: "u32",

		/**
		 * Optional feature flags for the whole compiled schema.
		 */
		flags: "u32",
	},

	// ---------------------------------------------------------------------
	// Schema node
	// ---------------------------------------------------------------------

	/**
	 * One node in the flattened schema graph.
	 *
	 * Every semantic schema type has one SchemaNode.
	 *
	 * The payload is intentionally fixed-size. Interpretation of a/b/c/d
	 * depends on `kind`.
	 */
	SchemaNode: {
		kind: "SchemaNodeKind",

		/**
		 * Kind-specific flags.
		 *
		 * Examples:
		 * - nullable
		 * - signed
		 * - bounded
		 * - exactLength
		 */
		flags: "u16",

		/**
		 * Kind-specific compact value.
		 *
		 * Usually:
		 * - primitive subtype
		 * - number of immediate children when <= 255
		 * - LiteralKind
		 */
		subkind: "u8",

		reserved: "u8",

		/**
		 * Kind-specific payload.
		 *
		 * See mapping below.
		 */
		a: "u32",
		b: "u32",
		c: "u32",
		d: "u32",
	},

	// ---------------------------------------------------------------------
	// Objects
	// ---------------------------------------------------------------------

	/**
	 * Object property definition.
	 *
	 * Object fields belonging to a node are contiguous inside
	 * ObjectFields[].
	 */
	ObjectField: {
		/**
		 * SchemaNode describing the property's value.
		 */
		node: "u16",

		/**
		 * Property flags:
		 *
		 * bit 0: required
		 * bit 1: nullable
		 * bit 2+: reserved
		 */
		flags: "u8",

		reserved: "u8",

		/**
		 * Tokenized property identifier.
		 *
		 * IMPORTANT:
		 * This span contains the semantic identifier itself:
		 *
		 *   temperature
		 *
		 * rather than JSON punctuation:
		 *
		 *   "temperature":
		 *
		 * Formatting tokens can be handled separately by the grammar.
		 */
		name: "TokenSpan",
	},

	// ---------------------------------------------------------------------
	// Enums
	// ---------------------------------------------------------------------

	/**
	 * One enum alternative.
	 *
	 * For constrained generation, `tokens` is the canonical representation.
	 *
	 * NumericValue can still be retained for binary-schema semantics, but
	 * token matching never needs to turn this back into text.
	 */
	EnumVariant: {
		tokens: "TokenSpan",

		/**
		 * Original enum ordinal / numeric representation.
		 *
		 * For string-only JSON schemas this can simply be the enum-local
		 * ordinal 0..N-1.
		 */
		value: "i32",
	},

	// ---------------------------------------------------------------------
	// Unions
	// ---------------------------------------------------------------------

	/**
	 * One union branch.
	 */
	UnionVariant: {
		/** SchemaNode representing the variant. */
		node: "u16",

		/**
		 * Numeric tag used by the compiled representation.
		 */
		tag: "u16",

		/**
		 * Token span for a symbolic discriminant value.
		 *
		 * tokenLength == 0 means that this variant has no tokenized
		 * discriminant.
		 */
		discriminant: "TokenSpan",
	},

	// ---------------------------------------------------------------------
	// Literals
	// ---------------------------------------------------------------------

	/**
	 * Literal payload table.
	 *
	 * It is tempting to put this directly in SchemaNode, but keeping a
	 * dedicated representation avoids forcing every future literal type
	 * into the generic a/b/c/d interpretation.
	 *
	 * String literals are represented by TokenSpan.
	 * Integer/boolean literals use valueLo/valueHi.
	 * Floats may store IEEE bits.
	 */
	LiteralValue: {
		kind: "LiteralKind",

		reserved0: "u8",
		reserved1: "u16",

		tokens: "TokenSpan",

		valueLo: "u32",
		valueHi: "u32",
	},

	// ---------------------------------------------------------------------
	// Optional numeric constraint tables
	// ---------------------------------------------------------------------

	/**
	 * Integer constraints.
	 *
	 * 64-bit values are represented as lo/hi pairs so WGSL does not require
	 * native i64/u64 support.
	 */
	IntegerConstraint: {
		minLo: "u32",
		minHi: "u32",

		maxLo: "u32",
		maxHi: "u32",

		/**
		 * bit 0: hasMin
		 * bit 1: hasMax
		 * bit 2: minExclusive
		 * bit 3: maxExclusive
		 */
		flags: "u32",
	},

	/**
	 * f32 constraints for the first version.
	 *
	 * If f64 genuinely becomes necessary, it can get its own table later
	 * rather than infecting the entire representation now.
	 */
	NumberConstraint: {
		min: "f32",
		max: "f32",

		/**
		 * bit 0: hasMin
		 * bit 1: hasMax
		 * bit 2: minExclusive
		 * bit 3: maxExclusive
		 */
		flags: "u32",

		reserved: "u32",
	},

	// ---------------------------------------------------------------------
	// String constraints
	// ---------------------------------------------------------------------

	StringConstraint: {
		minLength: "u32",
		maxLength: "u32",

		/**
		 * bit 0: hasMinLength
		 * bit 1: hasMaxLength
		 *
		 * Later:
		 * pattern, format, encoding classes, etc.
		 */
		flags: "u32",

		reserved: "u32",
	},

	// ---------------------------------------------------------------------
	// Array constraints
	// ---------------------------------------------------------------------

	ArrayConstraint: {
		/**
		 * SchemaNode of the item.
		 */
		itemNode: "u16",

		reserved: "u16",

		minItems: "u32",
		maxItems: "u32",

		/**
		 * bit 0: hasMinItems
		 * bit 1: hasMaxItems
		 * bit 2: uniqueItems
		 */
		flags: "u32",
	},

	// ---------------------------------------------------------------------
	// Token storage
	// ---------------------------------------------------------------------

	/**
	 * Physical token storage.
	 *
	 * This alias describes one packed word:
	 *
	 *   bits  0..15  = token N
	 *   bits 16..31  = token N+1
	 *
	 * The actual GPU buffer is an array of these.
	 */
	PackedSchemaTokens: "u32",
});