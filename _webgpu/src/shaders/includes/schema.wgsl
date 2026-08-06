// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

const SMB_prefill: u32 = 0u;
const SMB_decode: u32 = 1u;
const SMB_continuation: u32 = 2u;
const SMB_idle: u32 = 3u;
const SMB_running: u32 = 4u;
const SMB_eos: u32 = 5u;
const SMB_done: u32 = 6u;
const SMB_error: u32 = 7u;

alias OpParamsMode = u32;
const OpParamsMode_prefill: OpParamsMode = SMB_prefill;
const OpParamsMode_decode: OpParamsMode = SMB_decode;
const OpParamsMode_continuation: OpParamsMode = SMB_continuation;

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

alias LlmRuntimeStatus = u32;
const LlmRuntimeStatus_idle: LlmRuntimeStatus = SMB_idle;
const LlmRuntimeStatus_running: LlmRuntimeStatus = SMB_running;
const LlmRuntimeStatus_eos: LlmRuntimeStatus = SMB_eos;
const LlmRuntimeStatus_done: LlmRuntimeStatus = SMB_done;
const LlmRuntimeStatus_error: LlmRuntimeStatus = SMB_error;

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


