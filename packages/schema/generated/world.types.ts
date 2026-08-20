// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

export namespace v1_0_0 {
	export const PerceptOperandTag = {
		PerceptInstanceRef: 0,
		PerceptIntentRef: 1,
		PerceptSomethingRef: 2,
		PerceptSymbolRef: 3,
		PerceptUnknownRef: 4,
	} as const;
	export type PerceptOperandTag = "PerceptInstanceRef" | "PerceptIntentRef" | "PerceptSomethingRef" | "PerceptSymbolRef" | "PerceptUnknownRef";
	
	export const RelationRole = {
		agent: 0,
		patient: 1,
		instrument: 2,
		location: 3,
		time: 4,
		reason: 5,
	} as const;
	export type RelationRole = "agent" | "patient" | "instrument" | "location" | "time" | "reason";
	
	export interface WorldChannel {
		symbol: string;
		quota?: number;
	}
	
	export const WorldQuantityKind = {
		signed: 0,
		unipolar: 1,
		count: 2,
		proportion: 3,
	} as const;
	export type WorldQuantityKind = "signed" | "unipolar" | "count" | "proportion";
	
	export interface WorldQuantity {
		field: string;
		kind: WorldQuantityKind;
		polarity?: { negative: string; positive: string };
		of?: string;
	}
	
	export interface WorldSymbol {
		symbol: string;
		tokenId: number;
		tokenClass: string;
		flags?: number;
		arity?: number;
		semanticTypeToken?: number;
		inverseToken?: number;
	}
	
	export interface WorldRelationRole {
		role: RelationRole;
	}
	
	export const WorldRelationDomain = {
		external: 0,
		perceptual: 1,
		internal: 2,
		communicative: 3,
		postural: 4,
	} as const;
	export type WorldRelationDomain = "external" | "perceptual" | "internal" | "communicative" | "postural";
	
	export interface WorldRelation {
		relation: string;
		domain?: WorldRelationDomain;
		roles: WorldRelationRole[];
	}
	
	export const WorldVocabularyContract = {
		"krystal-world@3": 0,
	} as const;
	export type WorldVocabularyContract = "krystal-world@3";
	
	export interface WorldVocabulary {
		contract: WorldVocabularyContract;
		symbols: WorldSymbol[];
		channels: WorldChannel[];
		quantities: WorldQuantity[];
		relations: WorldRelation[];
	}
	
	export interface PerceptQuantity {
		field: string;
		value: number;
	}
	
	export const PerceptRecordEmptiness = {
		void: 0,
		unavailable: 1,
	} as const;
	export type PerceptRecordEmptiness = "void" | "unavailable";
	
	export interface PerceptRecord {
		channel: string;
		schema: string;
		instanceId?: string;
		tokens: string[];
		quantities?: PerceptQuantity[];
		count?: number;
		salience?: number;
		observedAt: number;
		emptiness?: PerceptRecordEmptiness;
	}
	
	export const PerceptInstanceRefKind = {
		instance: 0,
	} as const;
	export type PerceptInstanceRefKind = "instance";
	
	export interface PerceptInstanceRef {
		kind: PerceptInstanceRefKind;
		instanceId: string;
	}
	
	export const PerceptSymbolRefKind = {
		symbol: 0,
	} as const;
	export type PerceptSymbolRefKind = "symbol";
	
	export interface PerceptSymbolRef {
		kind: PerceptSymbolRefKind;
		symbol: string;
	}
	
	export const PerceptUnknownRefKind = {
		unknown: 0,
	} as const;
	export type PerceptUnknownRefKind = "unknown";
	
	export interface PerceptUnknownRef {
		kind: PerceptUnknownRefKind;
	}
	
	export const PerceptSomethingRefKind = {
		something: 0,
	} as const;
	export type PerceptSomethingRefKind = "something";
	
	export interface PerceptSomethingRef {
		kind: PerceptSomethingRefKind;
	}
	
	export const PerceptIntentRefKind = {
		intent: 0,
	} as const;
	export type PerceptIntentRefKind = "intent";
	
	export interface PerceptIntentRef {
		kind: PerceptIntentRefKind;
		intentRef: number;
	}
	
	export type PerceptOperand = ({ kind: "instance" } & PerceptInstanceRef) | ({ kind: "intent" } & PerceptIntentRef) | ({ kind: "something" } & PerceptSomethingRef) | ({ kind: "symbol" } & PerceptSymbolRef) | ({ kind: "unknown" } & PerceptUnknownRef);
	
	export interface PerceptRoleBinding {
		role: RelationRole;
		operand: PerceptOperand;
	}
	
	export const PerceptRelationAspect = {
		event: 0,
		state: 1,
	} as const;
	export type PerceptRelationAspect = "event" | "state";
	
	export interface PerceptRelation {
		channel: string;
		relation: string;
		roles: PerceptRoleBinding[];
		aspect: PerceptRelationAspect;
		quantities?: PerceptQuantity[];
		salience?: number;
		observedAt: number;
	}
	
	export const PerceptContract = {
		"krystal-percept@3": 0,
	} as const;
	export type PerceptContract = "krystal-percept@3";
	
	export interface Percept {
		contract: PerceptContract;
		tick: number;
		deltaMillis: number;
		valence: number;
		actorId: string;
		records: PerceptRecord[];
		relations?: PerceptRelation[];
	}
	
	export const LessonContract = {
		"krystal-lesson@3": 0,
	} as const;
	export type LessonContract = "krystal-lesson@3";
	
	export interface Lesson {
		contract: LessonContract;
		percept: Percept;
		expect: { relation: string; roles: PerceptRoleBinding[] };
		label?: string;
	}
	
	export interface AgentIntent {
		relation: string;
		roles: PerceptRoleBinding[];
		intensity: number;
		commitment: number;
		intentRef: number;
		volitive: boolean;
	}
	
	}
