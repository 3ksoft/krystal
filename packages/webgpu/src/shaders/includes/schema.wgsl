// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

alias OpParamsMode = u32;
const OpParamsMode_prefill: OpParamsMode = 0u;
const OpParamsMode_decode: OpParamsMode = 1u;
const OpParamsMode_continuation: OpParamsMode = 2u;

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
	mode: OpParamsMode,
	f0: f32,
	f1: f32,
	u0: u32,
	u1: u32,
	reserved: array<u32, 48>,
};

const OP_PARAMS_RESERVED_LEN: u32 = 48u;

struct Weights {
	f32: f32,
	u32: u32,
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


