// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

// THIS FILE IS FOR REFERENCE ONLY!! DO NOT INCCLUDE IT DIRECTLY!!
const SMB_literal: u32 = 0u;
const SMB_switch: u32 = 1u;
const SMB_string: u32 = 2u;
const SMB_number: u32 = 3u;
const SMB_accept: u32 = 4u;
const SMB_jump: u32 = 5u;
const SMB_running: u32 = 6u;
const SMB_dead: u32 = 7u;
const SMB_error: u32 = 8u;
const SMB_prefill: u32 = 9u;
const SMB_decode: u32 = 10u;
const SMB_continuation: u32 = 11u;
const SMB_idle: u32 = 12u;
const SMB_eos: u32 = 13u;
const SMB_done: u32 = 14u;
const SMB_f16: u32 = 15u;
const SMB_f32: u32 = 16u;
const SMB_wq4: u32 = 17u;
const SMB_conv: u32 = 18u;
const SMB_attention: u32 = 19u;

alias ConstraintNodeKind = u32;
const ConstraintNodeKind_literal: ConstraintNodeKind = 0u;
const ConstraintNodeKind_switch: ConstraintNodeKind = 1u;
const ConstraintNodeKind_string: ConstraintNodeKind = 2u;
const ConstraintNodeKind_number: ConstraintNodeKind = 3u;
const ConstraintNodeKind_accept: ConstraintNodeKind = 4u;
const ConstraintNodeKind_jump: ConstraintNodeKind = 5u;

alias ConstraintDecoderStatus = u32;
const ConstraintDecoderStatus_running: ConstraintDecoderStatus = 0u;
const ConstraintDecoderStatus_accept: ConstraintDecoderStatus = 1u;
const ConstraintDecoderStatus_dead: ConstraintDecoderStatus = 2u;
const ConstraintDecoderStatus_error: ConstraintDecoderStatus = 3u;

struct ConstraintProgramHeader {
	version: u32,
	flags: u32,
	entryNode: u32,
	acceptNode: u32,
	nodeWordOffset: u32,
	nodeCount: u32,
	edgeWordOffset: u32,
	edgeCount: u32,
	byteWordOffset: u32,
	byteLength: u32,
	reserved0: u32,
	reserved1: u32,
};


struct ConstraintNode {
	kind: ConstraintNodeKind,
	next: u32,
	dataOffset: u32,
	dataCount: u32,
	arg0: u32,
	arg1: u32,
	arg2: u32,
	arg3: u32,
	arg4: u32,
	arg5: u32,
	arg6: u32,
	arg7: u32,
};


struct ConstraintByteEdge {
	word: u32,
};


struct ConstraintTokenizerHeader {
	tokenCount: u32,
	eosToken: u32,
	entryWordOffset: u32,
	byteWordOffset: u32,
	byteLength: u32,
	reserved0: u32,
	reserved1: u32,
	reserved2: u32,
};


struct ConstraintTokenByteEntry {
	byteOffset: u32,
	meta: u32,
};


struct ConstraintDecoderState {
	node: u32,
	local0: u32,
	local1: u32,
	local2: u32,
	status: ConstraintDecoderStatus,
	errorCode: u32,
	reserved0: u32,
	reserved1: u32,
	numberText: array<u32, 16>,
};

const CONSTRAINT_DECODER_STATE_NUMBERTEXT_LEN: u32 = 16u;

alias Lfm2Mode = u32;
const Lfm2Mode_prefill: Lfm2Mode = 0u;
const Lfm2Mode_decode: Lfm2Mode = 1u;
const Lfm2Mode_continuation: Lfm2Mode = 2u;

struct OpParams {
	inputOffset: u32,
	outputOffset: u32,
	auxOffset: u32,
	aux2Offset: u32,
	tokenCount: u32,
	inputDim: u32,
	outputDim: u32,
	rowStart: u32,
	rowCount: u32,
	layerIndex: u32,
	attentionSlot: u32,
	mode: Lfm2Mode,
	f0: f32,
	f1: f32,
	u0: u32,
	u1: u32,
};


alias LlmRuntimeStatus = u32;
const LlmRuntimeStatus_idle: LlmRuntimeStatus = 0u;
const LlmRuntimeStatus_running: LlmRuntimeStatus = 1u;
const LlmRuntimeStatus_eos: LlmRuntimeStatus = 2u;
const LlmRuntimeStatus_done: LlmRuntimeStatus = 3u;
const LlmRuntimeStatus_error: LlmRuntimeStatus = 4u;

struct LlmRuntime {
	contextCapacity: u32,
	maxNewTokens: u32,
	eosToken: u32,
	promptTokenCount: u32,
	position: u32,
	generatedCount: u32,
	currentToken: u32,
	status: LlmRuntimeStatus,
	telemetryRevision: u32,
	lastToken: u32,
	errorCode: u32,
	pad0: u32,
};


struct DecodeTelemetryEntry {
	position: u32,
	status: u32,
	tokenId: u32,
};


struct GenerateOptions {
	maxNewTokens: u32,
	profile: u32,
};


struct GenerateTimings {
	prefillMs: u32,
	decodeMs: u32,
	readbackMs: u32,
	totalMs: u32,
	promptTokens: u32,
	scheduledDecodeSteps: u32,
	cacheDepth: u32,
	cachedBlocks: u32,
	cachedTokens: u32,
	liveQueryTokens: u32,
	repairedTokens: u32,
};


struct GenerateResult {
	tokenIds: array<u32, 0>,
	state: LlmRuntime,
	timings: u32,
};


struct CacheBlockOptions {
	depth: u32,
};


alias GpuWeightFormat = u32;
const GpuWeightFormat_f16: GpuWeightFormat = 0u;
const GpuWeightFormat_f32: GpuWeightFormat = 1u;
const GpuWeightFormat_wq4: GpuWeightFormat = 2u;

alias Lfm2LayerKind = u32;
const Lfm2LayerKind_conv: Lfm2LayerKind = 0u;
const Lfm2LayerKind_attention: Lfm2LayerKind = 1u;

struct Lfm2RuntimeConfig {
	contextLength: u32,
	hiddenSize: u32,
	feedForwardSize: u32,
	attentionHeads: u32,
	kvHeadsByLayer: array<u32, 0>,
	headDim: u32,
	ropeTheta: u32,
	vocabSize: u32,
	convCacheLength: u32,
	normEpsilon: u32,
	eosToken: u32,
	blockCount: u32,
	layers: array<Lfm2LayerKind, 0>,
	attentionLayerSlots: array<u32, 0>,
};


struct MatmulDispatchArgs {
	rowCount: u32,
	tokenCount: u32,
	inputDim: u32,
	outputDim: u32,
};


struct GpuTensorPage {
	buffer: GPUBuffer,
	rowStart: u32,
	rowCount: u32,
	byteLength: u32,
};


struct GpuTensor {
	name: u32,
	format: GpuWeightFormat,
	dimensions: array<u32, 0>,
	pages: array<GpuTensorPage, 0>,
	byteLength: u32,
};


struct Lfm2RuntimeOptions {
	contextCapacity: u32,
	maxNewTokens: u32,
	matmulKernels: u32,
};



// ==========================================
// MEMORY HELPERS (Storage Buffers interop)
// ==========================================

fn unpack_words_to_constraint_program_header(raw: array<u32, 12>) -> ConstraintProgramHeader {
	var out: ConstraintProgramHeader;
	out.version = bitcast<u32>(raw[0u]);
	out.flags = bitcast<u32>(raw[1u]);
	out.entryNode = bitcast<u32>(raw[2u]);
	out.acceptNode = bitcast<u32>(raw[3u]);
	out.nodeWordOffset = bitcast<u32>(raw[4u]);
	out.nodeCount = bitcast<u32>(raw[5u]);
	out.edgeWordOffset = bitcast<u32>(raw[6u]);
	out.edgeCount = bitcast<u32>(raw[7u]);
	out.byteWordOffset = bitcast<u32>(raw[8u]);
	out.byteLength = bitcast<u32>(raw[9u]);
	out.reserved0 = bitcast<u32>(raw[10u]);
	out.reserved1 = bitcast<u32>(raw[11u]);
	return out;
}

fn unpack_constraint_program_header(raw: array<u32, 12>) -> ConstraintProgramHeader {
	return unpack_words_to_constraint_program_header(raw);
}

fn pack_constraint_program_header_to_words(unpacked: ConstraintProgramHeader) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.version);
	out[1u] = bitcast<u32>(unpacked.flags);
	out[2u] = bitcast<u32>(unpacked.entryNode);
	out[3u] = bitcast<u32>(unpacked.acceptNode);
	out[4u] = bitcast<u32>(unpacked.nodeWordOffset);
	out[5u] = bitcast<u32>(unpacked.nodeCount);
	out[6u] = bitcast<u32>(unpacked.edgeWordOffset);
	out[7u] = bitcast<u32>(unpacked.edgeCount);
	out[8u] = bitcast<u32>(unpacked.byteWordOffset);
	out[9u] = bitcast<u32>(unpacked.byteLength);
	out[10u] = bitcast<u32>(unpacked.reserved0);
	out[11u] = bitcast<u32>(unpacked.reserved1);
	return out;
}

fn pack_constraint_program_header(unpacked: ConstraintProgramHeader) -> array<u32, 12> {
	return pack_constraint_program_header_to_words(unpacked);
}

fn unpack_words_to_constraint_node(raw: array<u32, 12>) -> ConstraintNode {
	var out: ConstraintNode;
	out.kind = ConstraintNodeKind(extractBits(raw[0u], 0u, 8u));
	out.next = bitcast<u32>(raw[1u]);
	out.dataOffset = bitcast<u32>(raw[2u]);
	out.dataCount = bitcast<u32>(raw[3u]);
	out.arg0 = bitcast<u32>(raw[4u]);
	out.arg1 = bitcast<u32>(raw[5u]);
	out.arg2 = bitcast<u32>(raw[6u]);
	out.arg3 = bitcast<u32>(raw[7u]);
	out.arg4 = bitcast<u32>(raw[8u]);
	out.arg5 = bitcast<u32>(raw[9u]);
	out.arg6 = bitcast<u32>(raw[10u]);
	out.arg7 = bitcast<u32>(raw[11u]);
	return out;
}

fn unpack_constraint_node(raw: array<u32, 12>) -> ConstraintNode {
	return unpack_words_to_constraint_node(raw);
}

fn pack_constraint_node_to_words(unpacked: ConstraintNode) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], u32(unpacked.kind), 0u, 8u);
	out[1u] = bitcast<u32>(unpacked.next);
	out[2u] = bitcast<u32>(unpacked.dataOffset);
	out[3u] = bitcast<u32>(unpacked.dataCount);
	out[4u] = bitcast<u32>(unpacked.arg0);
	out[5u] = bitcast<u32>(unpacked.arg1);
	out[6u] = bitcast<u32>(unpacked.arg2);
	out[7u] = bitcast<u32>(unpacked.arg3);
	out[8u] = bitcast<u32>(unpacked.arg4);
	out[9u] = bitcast<u32>(unpacked.arg5);
	out[10u] = bitcast<u32>(unpacked.arg6);
	out[11u] = bitcast<u32>(unpacked.arg7);
	return out;
}

fn pack_constraint_node(unpacked: ConstraintNode) -> array<u32, 12> {
	return pack_constraint_node_to_words(unpacked);
}

fn unpack_words_to_constraint_byte_edge(raw: u32) -> ConstraintByteEdge {
	var out: ConstraintByteEdge;
	out.word = bitcast<u32>(raw);
	return out;
}

fn unpack_constraint_byte_edge(raw: u32) -> ConstraintByteEdge {
	return unpack_words_to_constraint_byte_edge(raw);
}

fn pack_constraint_byte_edge_to_words(unpacked: ConstraintByteEdge) -> u32 {
	var out: u32 = 0u;
	out = bitcast<u32>(unpacked.word);
	return out;
}

fn pack_constraint_byte_edge(unpacked: ConstraintByteEdge) -> u32 {
	return pack_constraint_byte_edge_to_words(unpacked);
}

fn unpack_words_to_constraint_tokenizer_header(raw: array<u32, 8>) -> ConstraintTokenizerHeader {
	var out: ConstraintTokenizerHeader;
	out.tokenCount = bitcast<u32>(raw[0u]);
	out.eosToken = bitcast<u32>(raw[1u]);
	out.entryWordOffset = bitcast<u32>(raw[2u]);
	out.byteWordOffset = bitcast<u32>(raw[3u]);
	out.byteLength = bitcast<u32>(raw[4u]);
	out.reserved0 = bitcast<u32>(raw[5u]);
	out.reserved1 = bitcast<u32>(raw[6u]);
	out.reserved2 = bitcast<u32>(raw[7u]);
	return out;
}

fn unpack_constraint_tokenizer_header(raw: array<u32, 8>) -> ConstraintTokenizerHeader {
	return unpack_words_to_constraint_tokenizer_header(raw);
}

fn pack_constraint_tokenizer_header_to_words(unpacked: ConstraintTokenizerHeader) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.tokenCount);
	out[1u] = bitcast<u32>(unpacked.eosToken);
	out[2u] = bitcast<u32>(unpacked.entryWordOffset);
	out[3u] = bitcast<u32>(unpacked.byteWordOffset);
	out[4u] = bitcast<u32>(unpacked.byteLength);
	out[5u] = bitcast<u32>(unpacked.reserved0);
	out[6u] = bitcast<u32>(unpacked.reserved1);
	out[7u] = bitcast<u32>(unpacked.reserved2);
	return out;
}

fn pack_constraint_tokenizer_header(unpacked: ConstraintTokenizerHeader) -> array<u32, 8> {
	return pack_constraint_tokenizer_header_to_words(unpacked);
}

fn unpack_words_to_constraint_token_byte_entry(raw: array<u32, 2>) -> ConstraintTokenByteEntry {
	var out: ConstraintTokenByteEntry;
	out.byteOffset = bitcast<u32>(raw[0u]);
	out.meta = bitcast<u32>(raw[1u]);
	return out;
}

fn unpack_constraint_token_byte_entry(raw: array<u32, 2>) -> ConstraintTokenByteEntry {
	return unpack_words_to_constraint_token_byte_entry(raw);
}

fn pack_constraint_token_byte_entry_to_words(unpacked: ConstraintTokenByteEntry) -> array<u32, 2> {
	var out: array<u32, 2>;
	for (var w = 0u; w < 2u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.byteOffset);
	out[1u] = bitcast<u32>(unpacked.meta);
	return out;
}

fn pack_constraint_token_byte_entry(unpacked: ConstraintTokenByteEntry) -> array<u32, 2> {
	return pack_constraint_token_byte_entry_to_words(unpacked);
}

fn unpack_words_to_constraint_decoder_state(raw: array<u32, 24>) -> ConstraintDecoderState {
	var out: ConstraintDecoderState;
	out.node = bitcast<u32>(raw[0u]);
	out.local0 = bitcast<u32>(raw[1u]);
	out.local1 = bitcast<u32>(raw[2u]);
	out.local2 = bitcast<u32>(raw[3u]);
	out.status = ConstraintDecoderStatus(extractBits(raw[4u], 0u, 8u));
	out.errorCode = bitcast<u32>(raw[5u]);
	out.reserved0 = bitcast<u32>(raw[6u]);
	out.reserved1 = bitcast<u32>(raw[7u]);
	for (var i_0 = 0u; i_0 < 16u; i_0++) {
		out.numberText[i_0] = bitcast<u32>(raw[8u + (i_0 * 1u)]);
	}
	return out;
}

fn unpack_constraint_decoder_state(raw: array<u32, 24>) -> ConstraintDecoderState {
	return unpack_words_to_constraint_decoder_state(raw);
}

fn pack_constraint_decoder_state_to_words(unpacked: ConstraintDecoderState) -> array<u32, 24> {
	var out: array<u32, 24>;
	for (var w = 0u; w < 24u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.node);
	out[1u] = bitcast<u32>(unpacked.local0);
	out[2u] = bitcast<u32>(unpacked.local1);
	out[3u] = bitcast<u32>(unpacked.local2);
	out[4u] = insertBits(out[4u], u32(unpacked.status), 0u, 8u);
	out[5u] = bitcast<u32>(unpacked.errorCode);
	out[6u] = bitcast<u32>(unpacked.reserved0);
	out[7u] = bitcast<u32>(unpacked.reserved1);
	for (var i_0 = 0u; i_0 < 16u; i_0++) {
			out[8u + (i_0 * 1u)] = bitcast<u32>(unpacked.numberText[i_0]);
		}
	return out;
}

fn pack_constraint_decoder_state(unpacked: ConstraintDecoderState) -> array<u32, 24> {
	return pack_constraint_decoder_state_to_words(unpacked);
}

fn unpack_words_to_op_params(raw: array<u32, 16>) -> OpParams {
	var out: OpParams;
	out.inputOffset = bitcast<u32>(raw[0u]);
	out.outputOffset = bitcast<u32>(raw[1u]);
	out.auxOffset = bitcast<u32>(raw[2u]);
	out.aux2Offset = bitcast<u32>(raw[3u]);
	out.tokenCount = bitcast<u32>(raw[4u]);
	out.inputDim = bitcast<u32>(raw[5u]);
	out.outputDim = bitcast<u32>(raw[6u]);
	out.rowStart = bitcast<u32>(raw[7u]);
	out.rowCount = bitcast<u32>(raw[8u]);
	out.layerIndex = bitcast<u32>(raw[9u]);
	out.attentionSlot = bitcast<u32>(raw[10u]);
	out.mode = Lfm2Mode(extractBits(raw[11u], 0u, 8u));
	out.f0 = bitcast<f32>(raw[12u]);
	out.f1 = bitcast<f32>(raw[13u]);
	out.u0 = bitcast<u32>(raw[14u]);
	out.u1 = bitcast<u32>(raw[15u]);
	return out;
}

fn unpack_op_params(raw: array<u32, 16>) -> OpParams {
	return unpack_words_to_op_params(raw);
}

fn pack_op_params_to_words(unpacked: OpParams) -> array<u32, 16> {
	var out: array<u32, 16>;
	for (var w = 0u; w < 16u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.inputOffset);
	out[1u] = bitcast<u32>(unpacked.outputOffset);
	out[2u] = bitcast<u32>(unpacked.auxOffset);
	out[3u] = bitcast<u32>(unpacked.aux2Offset);
	out[4u] = bitcast<u32>(unpacked.tokenCount);
	out[5u] = bitcast<u32>(unpacked.inputDim);
	out[6u] = bitcast<u32>(unpacked.outputDim);
	out[7u] = bitcast<u32>(unpacked.rowStart);
	out[8u] = bitcast<u32>(unpacked.rowCount);
	out[9u] = bitcast<u32>(unpacked.layerIndex);
	out[10u] = bitcast<u32>(unpacked.attentionSlot);
	out[11u] = insertBits(out[11u], u32(unpacked.mode), 0u, 8u);
	out[12u] = bitcast<u32>(unpacked.f0);
	out[13u] = bitcast<u32>(unpacked.f1);
	out[14u] = bitcast<u32>(unpacked.u0);
	out[15u] = bitcast<u32>(unpacked.u1);
	return out;
}

fn pack_op_params(unpacked: OpParams) -> array<u32, 16> {
	return pack_op_params_to_words(unpacked);
}

fn unpack_words_to_llm_runtime(raw: array<u32, 12>) -> LlmRuntime {
	var out: LlmRuntime;
	out.contextCapacity = bitcast<u32>(raw[0u]);
	out.maxNewTokens = bitcast<u32>(raw[1u]);
	out.eosToken = bitcast<u32>(raw[2u]);
	out.promptTokenCount = bitcast<u32>(raw[3u]);
	out.position = bitcast<u32>(raw[4u]);
	out.generatedCount = bitcast<u32>(raw[5u]);
	out.currentToken = bitcast<u32>(raw[6u]);
	out.status = LlmRuntimeStatus(extractBits(raw[7u], 0u, 8u));
	out.telemetryRevision = bitcast<u32>(raw[8u]);
	out.lastToken = bitcast<u32>(raw[9u]);
	out.errorCode = bitcast<u32>(raw[10u]);
	out.pad0 = bitcast<u32>(raw[11u]);
	return out;
}

fn unpack_llm_runtime(raw: array<u32, 12>) -> LlmRuntime {
	return unpack_words_to_llm_runtime(raw);
}

fn pack_llm_runtime_to_words(unpacked: LlmRuntime) -> array<u32, 12> {
	var out: array<u32, 12>;
	for (var w = 0u; w < 12u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.contextCapacity);
	out[1u] = bitcast<u32>(unpacked.maxNewTokens);
	out[2u] = bitcast<u32>(unpacked.eosToken);
	out[3u] = bitcast<u32>(unpacked.promptTokenCount);
	out[4u] = bitcast<u32>(unpacked.position);
	out[5u] = bitcast<u32>(unpacked.generatedCount);
	out[6u] = bitcast<u32>(unpacked.currentToken);
	out[7u] = insertBits(out[7u], u32(unpacked.status), 0u, 8u);
	out[8u] = bitcast<u32>(unpacked.telemetryRevision);
	out[9u] = bitcast<u32>(unpacked.lastToken);
	out[10u] = bitcast<u32>(unpacked.errorCode);
	out[11u] = bitcast<u32>(unpacked.pad0);
	return out;
}

fn pack_llm_runtime(unpacked: LlmRuntime) -> array<u32, 12> {
	return pack_llm_runtime_to_words(unpacked);
}

fn unpack_words_to_decode_telemetry_entry(raw: u32) -> DecodeTelemetryEntry {
	var out: DecodeTelemetryEntry;
	out.position = u32(extractBits(raw, 0u, 8u));
	out.status = u32(extractBits(raw, 8u, 4u));
	out.tokenId = u32(extractBits(raw, 16u, 16u));
	return out;
}

fn unpack_decode_telemetry_entry(raw: u32) -> DecodeTelemetryEntry {
	return unpack_words_to_decode_telemetry_entry(raw);
}

fn pack_decode_telemetry_entry_to_words(unpacked: DecodeTelemetryEntry) -> u32 {
	var out: u32 = 0u;
	out = insertBits(out, u32(unpacked.position), 0u, 8u);
	out = insertBits(out, u32(unpacked.status), 8u, 4u);
	out = insertBits(out, u32(unpacked.tokenId), 16u, 16u);
	return out;
}

fn pack_decode_telemetry_entry(unpacked: DecodeTelemetryEntry) -> u32 {
	return pack_decode_telemetry_entry_to_words(unpacked);
}

fn unpack_words_to_generate_options(raw: array<u32, 5>) -> GenerateOptions {
	var out: GenerateOptions;
			return out;
}

fn unpack_generate_options(raw: array<u32, 5>) -> GenerateOptions {
	return unpack_words_to_generate_options(raw);
}

fn pack_generate_options_to_words(unpacked: GenerateOptions) -> array<u32, 5> {
	var out: array<u32, 5>;
	for (var w = 0u; w < 5u; w++) { out[w] = 0u; }
			return out;
}

fn pack_generate_options(unpacked: GenerateOptions) -> array<u32, 5> {
	return pack_generate_options_to_words(unpacked);
}

fn unpack_words_to_generate_timings(raw: array<u32, 27>) -> GenerateTimings {
	var out: GenerateTimings;
	out.prefillMs = bitcast<u32>(raw[0u]);
	out.decodeMs = bitcast<u32>(raw[2u]);
	out.readbackMs = bitcast<u32>(raw[4u]);
	out.totalMs = bitcast<u32>(raw[6u]);
	out.promptTokens = bitcast<u32>(raw[8u]);
	out.scheduledDecodeSteps = bitcast<u32>(raw[10u]);
						return out;
}

fn unpack_generate_timings(raw: array<u32, 27>) -> GenerateTimings {
	return unpack_words_to_generate_timings(raw);
}

fn pack_generate_timings_to_words(unpacked: GenerateTimings) -> array<u32, 27> {
	var out: array<u32, 27>;
	for (var w = 0u; w < 27u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.prefillMs);
	out[2u] = bitcast<u32>(unpacked.decodeMs);
	out[4u] = bitcast<u32>(unpacked.readbackMs);
	out[6u] = bitcast<u32>(unpacked.totalMs);
	out[8u] = bitcast<u32>(unpacked.promptTokens);
	out[10u] = bitcast<u32>(unpacked.scheduledDecodeSteps);
						return out;
}

fn pack_generate_timings(unpacked: GenerateTimings) -> array<u32, 27> {
	return pack_generate_timings_to_words(unpacked);
}

fn unpack_words_to_generate_result(raw: array<u32, 41>) -> GenerateResult {
	var out: GenerateResult;
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.tokenIds[i_0] = bitcast<u32>(raw[0u + (i_0 * 1u)]);
	}
	{
		var tmp: array<u32, 12>;
		for (var j_0 = 0u; j_0 < 12u; j_0++) { tmp[j_0] = raw[1u + j_0]; }
		out.state = unpack_words_to_llm_runtime(tmp);
	}
		return out;
}

fn unpack_generate_result(raw: array<u32, 41>) -> GenerateResult {
	return unpack_words_to_generate_result(raw);
}

fn pack_generate_result_to_words(unpacked: GenerateResult) -> array<u32, 41> {
	var out: array<u32, 41>;
	for (var w = 0u; w < 41u; w++) { out[w] = 0u; }
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[0u + (i_0 * 1u)] = bitcast<u32>(unpacked.tokenIds[i_0]);
		}
	{
		let tmp = pack_llm_runtime_to_words(unpacked.state);
		for (var j_0 = 0u; j_0 < 12u; j_0++) { out[1u + j_0] = tmp[j_0]; }
	}
		return out;
}

fn pack_generate_result(unpacked: GenerateResult) -> array<u32, 41> {
	return pack_generate_result_to_words(unpacked);
}

fn unpack_words_to_cache_block_options(raw: array<u32, 3>) -> CacheBlockOptions {
	var out: CacheBlockOptions;
		return out;
}

fn unpack_cache_block_options(raw: array<u32, 3>) -> CacheBlockOptions {
	return unpack_words_to_cache_block_options(raw);
}

fn pack_cache_block_options_to_words(unpacked: CacheBlockOptions) -> array<u32, 3> {
	var out: array<u32, 3>;
	for (var w = 0u; w < 3u; w++) { out[w] = 0u; }
		return out;
}

fn pack_cache_block_options(unpacked: CacheBlockOptions) -> array<u32, 3> {
	return pack_cache_block_options_to_words(unpacked);
}

fn unpack_words_to_lfm2_runtime_config(raw: array<u32, 25>) -> Lfm2RuntimeConfig {
	var out: Lfm2RuntimeConfig;
	out.contextLength = bitcast<u32>(raw[0u]);
	out.hiddenSize = bitcast<u32>(raw[2u]);
	out.feedForwardSize = bitcast<u32>(raw[4u]);
	out.attentionHeads = bitcast<u32>(raw[6u]);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.kvHeadsByLayer[i_0] = bitcast<u32>(raw[8u + (i_0 * 1u)]);
	}
	out.headDim = bitcast<u32>(raw[9u]);
	out.ropeTheta = bitcast<u32>(raw[11u]);
	out.vocabSize = bitcast<u32>(raw[13u]);
	out.convCacheLength = bitcast<u32>(raw[15u]);
	out.normEpsilon = bitcast<u32>(raw[17u]);
	out.eosToken = bitcast<u32>(raw[19u]);
	out.blockCount = bitcast<u32>(raw[21u]);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.layers[i_0] = Lfm2LayerKind(raw[23u + (i_0 * 1u)]);
	}
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.attentionLayerSlots[i_0] = bitcast<u32>(raw[24u + (i_0 * 1u)]);
	}
	return out;
}

fn unpack_lfm2_runtime_config(raw: array<u32, 25>) -> Lfm2RuntimeConfig {
	return unpack_words_to_lfm2_runtime_config(raw);
}

fn pack_lfm2_runtime_config_to_words(unpacked: Lfm2RuntimeConfig) -> array<u32, 25> {
	var out: array<u32, 25>;
	for (var w = 0u; w < 25u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.contextLength);
	out[2u] = bitcast<u32>(unpacked.hiddenSize);
	out[4u] = bitcast<u32>(unpacked.feedForwardSize);
	out[6u] = bitcast<u32>(unpacked.attentionHeads);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[8u + (i_0 * 1u)] = bitcast<u32>(unpacked.kvHeadsByLayer[i_0]);
		}
	out[9u] = bitcast<u32>(unpacked.headDim);
	out[11u] = bitcast<u32>(unpacked.ropeTheta);
	out[13u] = bitcast<u32>(unpacked.vocabSize);
	out[15u] = bitcast<u32>(unpacked.convCacheLength);
	out[17u] = bitcast<u32>(unpacked.normEpsilon);
	out[19u] = bitcast<u32>(unpacked.eosToken);
	out[21u] = bitcast<u32>(unpacked.blockCount);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[23u + (i_0 * 1u)] = u32(unpacked.layers[i_0]);
		}
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[24u + (i_0 * 1u)] = bitcast<u32>(unpacked.attentionLayerSlots[i_0]);
		}
	return out;
}

fn pack_lfm2_runtime_config(unpacked: Lfm2RuntimeConfig) -> array<u32, 25> {
	return pack_lfm2_runtime_config_to_words(unpacked);
}

fn unpack_words_to_matmul_dispatch_args(raw: array<u32, 8>) -> MatmulDispatchArgs {
	var out: MatmulDispatchArgs;
	out.rowCount = bitcast<u32>(raw[0u]);
	out.tokenCount = bitcast<u32>(raw[2u]);
	out.inputDim = bitcast<u32>(raw[4u]);
	out.outputDim = bitcast<u32>(raw[6u]);
	return out;
}

fn unpack_matmul_dispatch_args(raw: array<u32, 8>) -> MatmulDispatchArgs {
	return unpack_words_to_matmul_dispatch_args(raw);
}

fn pack_matmul_dispatch_args_to_words(unpacked: MatmulDispatchArgs) -> array<u32, 8> {
	var out: array<u32, 8>;
	for (var w = 0u; w < 8u; w++) { out[w] = 0u; }
	out[0u] = bitcast<u32>(unpacked.rowCount);
	out[2u] = bitcast<u32>(unpacked.tokenCount);
	out[4u] = bitcast<u32>(unpacked.inputDim);
	out[6u] = bitcast<u32>(unpacked.outputDim);
	return out;
}

fn pack_matmul_dispatch_args(unpacked: MatmulDispatchArgs) -> array<u32, 8> {
	return pack_matmul_dispatch_args_to_words(unpacked);
}

fn unpack_words_to_gpu_buffer(raw: u32) -> GPUBuffer {
	var out: GPUBuffer;
		return out;
}

fn unpack_gpu_buffer(raw: u32) -> GPUBuffer {
	return unpack_words_to_gpu_buffer(raw);
}

fn pack_gpu_buffer_to_words(unpacked: GPUBuffer) -> u32 {
	var out: u32 = 0u;
		return out;
}

fn pack_gpu_buffer(unpacked: GPUBuffer) -> u32 {
	return pack_gpu_buffer_to_words(unpacked);
}

fn unpack_words_to_gpu_device(raw: u32) -> GPUDevice {
	var out: GPUDevice;
		return out;
}

fn unpack_gpu_device(raw: u32) -> GPUDevice {
	return unpack_words_to_gpu_device(raw);
}

fn pack_gpu_device_to_words(unpacked: GPUDevice) -> u32 {
	var out: u32 = 0u;
		return out;
}

fn pack_gpu_device(unpacked: GPUDevice) -> u32 {
	return pack_gpu_device_to_words(unpacked);
}

fn unpack_words_to_gpu_tensor_page(raw: array<u32, 6>) -> GpuTensorPage {
	var out: GpuTensorPage;
	out.buffer = unpack_words_to_gpu_buffer(extractBits(raw[0u], 0u, 0u));
	out.rowStart = bitcast<u32>(raw[0u]);
	out.rowCount = bitcast<u32>(raw[2u]);
	out.byteLength = bitcast<u32>(raw[4u]);
	return out;
}

fn unpack_gpu_tensor_page(raw: array<u32, 6>) -> GpuTensorPage {
	return unpack_words_to_gpu_tensor_page(raw);
}

fn pack_gpu_tensor_page_to_words(unpacked: GpuTensorPage) -> array<u32, 6> {
	var out: array<u32, 6>;
	for (var w = 0u; w < 6u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], pack_gpu_buffer_to_words(unpacked.buffer), 0u, 0u);
	out[0u] = bitcast<u32>(unpacked.rowStart);
	out[2u] = bitcast<u32>(unpacked.rowCount);
	out[4u] = bitcast<u32>(unpacked.byteLength);
	return out;
}

fn pack_gpu_tensor_page(unpacked: GpuTensorPage) -> array<u32, 6> {
	return pack_gpu_tensor_page_to_words(unpacked);
}

fn unpack_words_to_gpu_tensor(raw: array<u32, 7>) -> GpuTensor {
	var out: GpuTensor;
		out.format = GpuWeightFormat(extractBits(raw[1u], 0u, 8u));
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		out.dimensions[i_0] = bitcast<u32>(raw[2u + (i_0 * 1u)]);
	}
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
		{
		var tmp: array<u32, 6>;
		for (var j_1 = 0u; j_1 < 6u; j_1++) { tmp[j_1] = raw[3u + (i_0 * 6u) + j_1]; }
		out.pages[i_0] = unpack_words_to_gpu_tensor_page(tmp);
	}
	}
		return out;
}

fn unpack_gpu_tensor(raw: array<u32, 7>) -> GpuTensor {
	return unpack_words_to_gpu_tensor(raw);
}

fn pack_gpu_tensor_to_words(unpacked: GpuTensor) -> array<u32, 7> {
	var out: array<u32, 7>;
	for (var w = 0u; w < 7u; w++) { out[w] = 0u; }
		out[1u] = insertBits(out[1u], u32(unpacked.format), 0u, 8u);
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			out[2u + (i_0 * 1u)] = bitcast<u32>(unpacked.dimensions[i_0]);
		}
	for (var i_0 = 0u; i_0 < 0u; i_0++) {
			{
		let tmp = pack_gpu_tensor_page_to_words(unpacked.pages[i_0]);
		for (var j_1 = 0u; j_1 < 6u; j_1++) { out[3u + (i_0 * 6u) + j_1] = tmp[j_1]; }
	}
		}
		return out;
}

fn pack_gpu_tensor(unpacked: GpuTensor) -> array<u32, 7> {
	return pack_gpu_tensor_to_words(unpacked);
}

fn unpack_words_to_lfm2_runtime_model(raw: array<u32, 26>) -> Lfm2RuntimeModel {
	var out: Lfm2RuntimeModel;
	out.device = unpack_words_to_gpu_device(extractBits(raw[0u], 0u, 0u));
	{
		var tmp: array<u32, 25>;
		for (var j_0 = 0u; j_0 < 25u; j_0++) { tmp[j_0] = raw[0u + j_0]; }
		out.config = unpack_words_to_lfm2_runtime_config(tmp);
	}
	out.tensor = u32(extractBits(raw[25u], 0u, 8u));
	return out;
}

fn unpack_lfm2_runtime_model(raw: array<u32, 26>) -> Lfm2RuntimeModel {
	return unpack_words_to_lfm2_runtime_model(raw);
}

fn pack_lfm2_runtime_model_to_words(unpacked: Lfm2RuntimeModel) -> array<u32, 26> {
	var out: array<u32, 26>;
	for (var w = 0u; w < 26u; w++) { out[w] = 0u; }
	out[0u] = insertBits(out[0u], pack_gpu_device_to_words(unpacked.device), 0u, 0u);
	{
		let tmp = pack_lfm2_runtime_config_to_words(unpacked.config);
		for (var j_0 = 0u; j_0 < 25u; j_0++) { out[0u + j_0] = tmp[j_0]; }
	}
	out[25u] = insertBits(out[25u], u32(unpacked.tensor), 0u, 8u);
	return out;
}

fn pack_lfm2_runtime_model(unpacked: Lfm2RuntimeModel) -> array<u32, 26> {
	return pack_lfm2_runtime_model_to_words(unpacked);
}

fn unpack_words_to_matmul_kernel_spec(raw: array<u32, 4>) -> MatmulKernelSpec {
	var out: MatmulKernelSpec;
			out.workgroups = u32(extractBits(raw[3u], 0u, 8u));
	return out;
}

fn unpack_matmul_kernel_spec(raw: array<u32, 4>) -> MatmulKernelSpec {
	return unpack_words_to_matmul_kernel_spec(raw);
}

fn pack_matmul_kernel_spec_to_words(unpacked: MatmulKernelSpec) -> array<u32, 4> {
	var out: array<u32, 4>;
	for (var w = 0u; w < 4u; w++) { out[w] = 0u; }
			out[3u] = insertBits(out[3u], u32(unpacked.workgroups), 0u, 8u);
	return out;
}

fn pack_matmul_kernel_spec(unpacked: MatmulKernelSpec) -> array<u32, 4> {
	return pack_matmul_kernel_spec_to_words(unpacked);
}

fn unpack_words_to_lfm2_runtime_options(raw: array<u32, 22>) -> Lfm2RuntimeOptions {
	var out: Lfm2RuntimeOptions;
				return out;
}

fn unpack_lfm2_runtime_options(raw: array<u32, 22>) -> Lfm2RuntimeOptions {
	return unpack_words_to_lfm2_runtime_options(raw);
}

fn pack_lfm2_runtime_options_to_words(unpacked: Lfm2RuntimeOptions) -> array<u32, 22> {
	var out: array<u32, 22>;
	for (var w = 0u; w < 22u; w++) { out[w] = 0u; }
				return out;
}

fn pack_lfm2_runtime_options(unpacked: Lfm2RuntimeOptions) -> array<u32, 22> {
	return pack_lfm2_runtime_options_to_words(unpacked);
}

