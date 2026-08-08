/**
 * Reserved bit patterns and fixed ABI capacities shared by the compiled schema
 * and guide runtime.
 *
 * Pure-TS on purpose: this module carries no arktype/@schema-pop imports, so a
 * statically compiled runtime (the scriptc native exe) can read
 * GPU_SCHEMA_SENTINELS without dragging the GPU schema scope into the graph.
 * sparse.ts builds that scope; it re-exports these constants for compatibility.
 */

/**
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
