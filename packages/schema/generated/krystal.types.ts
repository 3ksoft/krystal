// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

export namespace v1_0_0 {
	export const BrainBandKind = {
		system: 0,
		homeostasis: 1,
		body: 2,
		perception: 3,
		memory: 4,
		focus: 5,
		query: 6,
		catalog: 7,
	} as const;
	export type BrainBandKind = "system" | "homeostasis" | "body" | "perception" | "memory" | "focus" | "query" | "catalog";
	
	export const BrainValueKind = {
		none: 0,
		token: 1,
		context_ref: 2,
		record_ref: 3,
		boolean_class: 4,
		scalar_band: 5,
		quantity_projection: 6,
		opaque_payload: 7,
	} as const;
	export type BrainValueKind = "none" | "token" | "context_ref" | "record_ref" | "boolean_class" | "scalar_band" | "quantity_projection" | "opaque_payload";
	
	export interface BrainFrameGpu {
		tokenIds: number[];
		fieldRoles: number[];
		schemaIds: number[];
		bandIds: number[];
		activeRecordIndices: number[];
	}
	
	}
