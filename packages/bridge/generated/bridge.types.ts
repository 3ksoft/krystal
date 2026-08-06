export namespace v1_0_0 {
	export const EngineEventTag = {
		Completed: 0,
		ExecutionStats: 1,
		Failed: 2,
		TokenEmitted: 3,
	} as const;
	export type EngineEventTag = "Completed" | "ExecutionStats" | "Failed" | "TokenEmitted";

	export const EngineCommandTag = {
		Cancel: 0,
		CreateCheckpoint: 1,
		DropBlock: 2,
		DropCheckpoint: 3,
		Generate: 4,
		PutBlock: 5,
	} as const;
	export type EngineCommandTag = "Cancel" | "CreateCheckpoint" | "DropBlock" | "DropCheckpoint" | "Generate" | "PutBlock";

	export const FrameDirection = {
		command: 0,
		event: 1,
	} as const;
	export type FrameDirection = "command" | "event";

	export interface FrameHeader {
		magic: number;
		version: number;
		direction: FrameDirection;
		flags: number;
		bodyBytes: number;
		payloadBytes: number;
	}

	export interface ContextRef {
		checkpoint: number;
		blockCount: number;
		reserved: number;
	}

	export interface PutBlock {
		operation: number;
		block: number;
		tokenCount: number;
	}

	export interface DropBlock {
		operation: number;
		block: number;
	}

	export interface CreateCheckpoint {
		operation: number;
		checkpoint: number;
		context: ContextRef;
	}

	export interface DropCheckpoint {
		operation: number;
		checkpoint: number;
	}

	export interface Generate {
		operation: number;
		context: ContextRef;
		maxTokens: number;
	}

	export interface Cancel {
		operation: number;
		target: number;
	}

	export type EngineCommand = ({ kind: "Cancel" } & Cancel) | ({ kind: "CreateCheckpoint" } & CreateCheckpoint) | ({ kind: "DropBlock" } & DropBlock) | ({ kind: "DropCheckpoint" } & DropCheckpoint) | ({ kind: "Generate" } & Generate) | ({ kind: "PutBlock" } & PutBlock);

	export const ErrorCode = {
		InvalidCommand: 0,
		InvalidContext: 1,
		NotFound: 2,
		CapacityExceeded: 3,
		Cancelled: 4,
		InternalError: 5,
	} as const;
	export type ErrorCode = "InvalidCommand" | "InvalidContext" | "NotFound" | "CapacityExceeded" | "Cancelled" | "InternalError";

	export interface Completed {
		operation: number;
	}

	export interface TokenEmitted {
		operation: number;
		token: number;
	}

	export interface ExecutionStats {
		operation: number;
		prefillTokens: number;
		checkpointHits: number;
		checkpointMisses: number;
		restoredBytes: number;
		checkpointBytes: number;
		kvBytes: number;
		kvCapacityBytes: number;
		convBytes: number;
		hiddenBytes: number;
		checkpointCreateUs: number;
		checkpointRestoreUs: number;
	}

	export interface Failed {
		operation: number;
		messageBytes: number;
		code: ErrorCode;
		reserved: number;
	}

	export type EngineEvent = ({ kind: "Completed" } & Completed) | ({ kind: "ExecutionStats" } & ExecutionStats) | ({ kind: "Failed" } & Failed) | ({ kind: "TokenEmitted" } & TokenEmitted);

	}
