// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

fn unpack_words_to_op_params(raw: array<u32, 64>) -> OpParams {
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
	out.mode = OpParamsMode(extractBits(raw[11u], 0u, 8u));
	out.f0 = bitcast<f32>(raw[12u]);
	out.f1 = bitcast<f32>(raw[13u]);
	out.u0 = bitcast<u32>(raw[14u]);
	out.u1 = bitcast<u32>(raw[15u]);
	for (var i_0 = 0u; i_0 < 48u; i_0++) {
		out.reserved[i_0] = bitcast<u32>(raw[16u + (i_0 * 1u)]);
	}
	return out;
}

fn unpack_op_params(raw: array<u32, 64>) -> OpParams {
	return unpack_words_to_op_params(raw);
}

fn pack_op_params_to_words(unpacked: OpParams) -> array<u32, 64> {
	var out: array<u32, 64>;
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
	out[11u] = 0u;
	out[11u] = insertBits(out[11u], u32(unpacked.mode), 0u, 8u);
	out[12u] = bitcast<u32>(unpacked.f0);
	out[13u] = bitcast<u32>(unpacked.f1);
	out[14u] = bitcast<u32>(unpacked.u0);
	out[15u] = bitcast<u32>(unpacked.u1);
	for (var i_0 = 0u; i_0 < 48u; i_0++) {
			out[16u + (i_0 * 1u)] = bitcast<u32>(unpacked.reserved[i_0]);
		}
	return out;
}

fn pack_op_params(unpacked: OpParams) -> array<u32, 64> {
	return pack_op_params_to_words(unpacked);
}

fn unpack_words_to_weights(raw: array<u32, 2>) -> Weights {
	var out: Weights;
	out.f32 = bitcast<f32>(raw[0u]);
	out.u32 = bitcast<u32>(raw[1u]);
	return out;
}

fn unpack_weights(raw: array<u32, 2>) -> Weights {
	return unpack_words_to_weights(raw);
}

fn pack_weights_to_words(unpacked: Weights) -> array<u32, 2> {
	var out: array<u32, 2>;
	out[0u] = bitcast<u32>(unpacked.f32);
	out[1u] = bitcast<u32>(unpacked.u32);
	return out;
}

fn pack_weights(unpacked: Weights) -> array<u32, 2> {
	return pack_weights_to_words(unpacked);
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
	out[0u] = bitcast<u32>(unpacked.contextCapacity);
	out[1u] = bitcast<u32>(unpacked.maxNewTokens);
	out[2u] = bitcast<u32>(unpacked.eosToken);
	out[3u] = bitcast<u32>(unpacked.promptTokenCount);
	out[4u] = bitcast<u32>(unpacked.position);
	out[5u] = bitcast<u32>(unpacked.generatedCount);
	out[6u] = bitcast<u32>(unpacked.currentToken);
	out[7u] = 0u;
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
	var out: u32;
	out = 0u;
	out = insertBits(out, u32(unpacked.position), 0u, 8u);
	out = insertBits(out, u32(unpacked.status), 8u, 4u);
	out = insertBits(out, u32(unpacked.tokenId), 16u, 16u);
	return out;
}

fn pack_decode_telemetry_entry(unpacked: DecodeTelemetryEntry) -> u32 {
	return pack_decode_telemetry_entry_to_words(unpacked);
}

