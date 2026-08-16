// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

export namespace v1_0_0 {
	export const ConstraintNodeKind = {
		literal: 0,
		switch: 1,
		string: 2,
		number: 3,
		accept: 4,
		jump: 5,
	} as const;
	export type ConstraintNodeKind = "literal" | "switch" | "string" | "number" | "accept" | "jump";
	
	export const ConstraintDecoderStatus = {
		running: 0,
		accept: 1,
		dead: 2,
		error: 3,
	} as const;
	export type ConstraintDecoderStatus = "running" | "accept" | "dead" | "error";
	
	export interface ConstraintProgramHeader {
		version: number;
		flags: number;
		entryNode: number;
		acceptNode: number;
		nodeWordOffset: number;
		nodeCount: number;
		edgeWordOffset: number;
		edgeCount: number;
		byteWordOffset: number;
		byteLength: number;
		reserved0: number;
		reserved1: number;
	}
	
	export interface ConstraintNode {
		kind: ConstraintNodeKind;
		next: number;
		dataOffset: number;
		dataCount: number;
		arg0: number;
		arg1: number;
		arg2: number;
		arg3: number;
		arg4: number;
		arg5: number;
		arg6: number;
		arg7: number;
	}
	
	export interface ConstraintByteEdge {
		word: number;
	}
	
	export interface ConstraintTokenizerHeader {
		tokenCount: number;
		eosToken: number;
		entryWordOffset: number;
		byteWordOffset: number;
		byteLength: number;
		reserved0: number;
		reserved1: number;
		reserved2: number;
	}
	
	export interface ConstraintTokenByteEntry {
		byteOffset: number;
		meta: number;
	}
	
	export interface ConstraintDecoderState {
		node: number;
		local0: number;
		local1: number;
		local2: number;
		status: ConstraintDecoderStatus;
		errorCode: number;
		reserved0: number;
		reserved1: number;
		numberText: number[];
	}
	
	export const Lfm2Mode = {
		prefill: 0,
		decode: 1,
		continuation: 2,
	} as const;
	export type Lfm2Mode = "prefill" | "decode" | "continuation";
	
	export interface OpParams {
		inputOffset: number;
		outputOffset: number;
		auxOffset: number;
		aux2Offset: number;
		aux3Offset: number;
		aux4Offset: number;
		aux5Offset: number;
		aux6Offset: number;
		tokenCount: number;
		inputDim: number;
		outputDim: number;
		rowStart: number;
		rowCount: number;
		layerIndex: number;
		attentionSlot: number;
		mode: Lfm2Mode;
		f0: number;
		f1: number;
		u0: number;
		u1: number;
		u2: number;
		u3: number;
		u4: number;
		u5: number;
	}
	
	export const LlmRuntimeStatus = {
		idle: 0,
		running: 1,
		eos: 2,
		done: 3,
		error: 4,
	} as const;
	export type LlmRuntimeStatus = "idle" | "running" | "eos" | "done" | "error";
	
	export interface LlmRuntime {
		contextCapacity: number;
		maxNewTokens: number;
		eosToken: number;
		promptTokenCount: number;
		position: number;
		generatedCount: number;
		currentToken: number;
		status: LlmRuntimeStatus;
		telemetryRevision: number;
		lastToken: number;
		errorCode: number;
		pad0: number;
	}
	
	export interface DecodeTelemetryEntry {
		position: number;
		status: number;
		tokenId: number;
	}
	
	export interface GenerateOptions {
		maxNewTokens?: number;
		profile?: boolean;
	}
	
	export interface GenerateTimings {
		prefillMs: number;
		decodeMs: number;
		readbackMs: number;
		totalMs: number;
		promptTokens: number;
		scheduledDecodeSteps: number;
		cacheDepth?: number;
		cachedBlocks?: number;
		cachedTokens?: number;
		liveQueryTokens?: number;
		repairedTokens?: number;
	}
	
	export interface GenerateResult {
		tokenIds: number[];
		state: LlmRuntime;
		timings?: GenerateTimings;
	}
	
	export interface CacheBlockOptions {
		depth?: number;
	}
	
	export const GpuWeightFormat = {
		f16: 0,
		f32: 1,
		wq4: 2,
	} as const;
	export type GpuWeightFormat = "f16" | "f32" | "wq4";
	
	export const Lfm2LayerKind = {
		conv: 0,
		attention: 1,
	} as const;
	export type Lfm2LayerKind = "conv" | "attention";
	
	export interface Lfm2RuntimeConfig {
		contextLength: number;
		hiddenSize: number;
		feedForwardSize: number;
		attentionHeads: number;
		kvHeadsByLayer: number[];
		headDim: number;
		ropeTheta: number;
		vocabSize: number;
		convCacheLength: number;
		normEpsilon: number;
		eosToken: number;
		blockCount: number;
		layers: Lfm2LayerKind[];
		attentionLayerSlots: number[];
	}
	
	export interface MatmulDispatchArgs {
		rowCount: number;
		tokenCount: number;
		inputDim: number;
		outputDim: number;
	}
	
	export type GPUBuffer = unknown;
	
	export type GPUDevice = unknown;
	
	export interface GpuTensorPage {
		buffer: GPUBuffer;
		rowStart: number;
		rowCount: number;
		byteLength: number;
	}
	
	export interface GpuTensor {
		name: string;
		format: GpuWeightFormat;
		dimensions: number[];
		pages: GpuTensorPage[];
		byteLength?: number;
	}
	
	export interface Lfm2RuntimeModel {
		device: GPUDevice;
		config: Lfm2RuntimeConfig;
		tensor: unknown;
	}
	
	export interface MatmulKernelSpec {
		entryPoint: string;
		wgsl?: string;
		workgroups?: unknown;
	}
	
	export interface Lfm2RuntimeOptions {
		contextCapacity?: number;
		maxNewTokens?: number;
		matmulKernels?: { f16: MatmulKernelSpec | undefined; f32: MatmulKernelSpec | undefined; wq4: MatmulKernelSpec | undefined };
	}
	
	}
