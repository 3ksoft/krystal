// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <array>

namespace v1_0_0 {
	using ConstraintNodeKind = uint8_t;
	constexpr ConstraintNodeKind ConstraintNodeKind_literal = 0;
	constexpr ConstraintNodeKind ConstraintNodeKind_switch = 1;
	constexpr ConstraintNodeKind ConstraintNodeKind_string = 2;
	constexpr ConstraintNodeKind ConstraintNodeKind_number = 3;
	constexpr ConstraintNodeKind ConstraintNodeKind_accept = 4;
	constexpr ConstraintNodeKind ConstraintNodeKind_jump = 5;
	
	using ConstraintDecoderStatus = uint8_t;
	constexpr ConstraintDecoderStatus ConstraintDecoderStatus_running = 0;
	constexpr ConstraintDecoderStatus ConstraintDecoderStatus_accept = 1;
	constexpr ConstraintDecoderStatus ConstraintDecoderStatus_dead = 2;
	constexpr ConstraintDecoderStatus ConstraintDecoderStatus_error = 3;
	
	struct alignas(4) ConstraintProgramHeader {
		uint32_t version;
		uint32_t flags;
		uint32_t entryNode;
		uint32_t acceptNode;
		uint32_t nodeWordOffset;
		uint32_t nodeCount;
		uint32_t edgeWordOffset;
		uint32_t edgeCount;
		uint32_t byteWordOffset;
		uint32_t byteLength;
		uint32_t reserved0;
		uint32_t reserved1;
	};
	
	struct alignas(4) ConstraintNode {
		ConstraintNodeKind kind;
		uint8_t _pad_kind[3];
		uint32_t next;
		uint32_t dataOffset;
		uint32_t dataCount;
		uint32_t arg0;
		uint32_t arg1;
		uint32_t arg2;
		uint32_t arg3;
		uint32_t arg4;
		uint32_t arg5;
		uint32_t arg6;
		uint32_t arg7;
	};
	
	struct alignas(4) ConstraintByteEdge {
		uint32_t word;
	};
	
	struct alignas(4) ConstraintTokenizerHeader {
		uint32_t tokenCount;
		uint32_t eosToken;
		uint32_t entryWordOffset;
		uint32_t byteWordOffset;
		uint32_t byteLength;
		uint32_t reserved0;
		uint32_t reserved1;
		uint32_t reserved2;
	};
	
	struct alignas(4) ConstraintTokenByteEntry {
		uint32_t byteOffset;
		uint32_t meta;
	};
	
	struct alignas(4) ConstraintDecoderState {
		uint32_t node;
		uint32_t local0;
		uint32_t local1;
		uint32_t local2;
		ConstraintDecoderStatus status;
		uint8_t _pad_status[3];
		uint32_t errorCode;
		uint32_t reserved0;
		uint32_t reserved1;
		uint8_t numberText[64];
	};
	
	using Lfm2Mode = uint8_t;
	constexpr Lfm2Mode Lfm2Mode_prefill = 0;
	constexpr Lfm2Mode Lfm2Mode_decode = 1;
	constexpr Lfm2Mode Lfm2Mode_continuation = 2;
	
	struct alignas(4) OpParams {
		uint32_t inputOffset;
		uint32_t outputOffset;
		uint32_t auxOffset;
		uint32_t aux2Offset;
		uint32_t tokenCount;
		uint32_t inputDim;
		uint32_t outputDim;
		uint32_t rowStart;
		uint32_t rowCount;
		uint32_t layerIndex;
		uint32_t attentionSlot;
		Lfm2Mode mode;
		uint8_t _pad_mode[3];
		float f0;
		float f1;
		uint32_t u0;
		uint32_t u1;
	};
	
	using LlmRuntimeStatus = uint8_t;
	constexpr LlmRuntimeStatus LlmRuntimeStatus_idle = 0;
	constexpr LlmRuntimeStatus LlmRuntimeStatus_running = 1;
	constexpr LlmRuntimeStatus LlmRuntimeStatus_eos = 2;
	constexpr LlmRuntimeStatus LlmRuntimeStatus_done = 3;
	constexpr LlmRuntimeStatus LlmRuntimeStatus_error = 4;
	
	struct alignas(4) LlmRuntime {
		uint32_t contextCapacity;
		uint32_t maxNewTokens;
		uint32_t eosToken;
		uint32_t promptTokenCount;
		uint32_t position;
		uint32_t generatedCount;
		uint32_t currentToken;
		LlmRuntimeStatus status;
		uint8_t _pad_status[3];
		uint32_t telemetryRevision;
		uint32_t lastToken;
		uint32_t errorCode;
		uint32_t pad0;
	};
	
	struct alignas(2) DecodeTelemetryEntry {
		uint8_t position;
		uint8_t status : 4;
		uint16_t tokenId;
	};
	
	struct alignas(4) GenerateOptions {
		uint8_t maxNewTokens[12];
		uint8_t profile[8];
	};
	
	struct alignas(4) GenerateTimings {
		double prefillMs;
		double decodeMs;
		double readbackMs;
		double totalMs;
		double promptTokens;
		double scheduledDecodeSteps;
		uint8_t cacheDepth[12];
		uint8_t cachedBlocks[12];
		uint8_t cachedTokens[12];
		uint8_t liveQueryTokens[12];
		uint8_t repairedTokens[12];
	};
	
	struct alignas(4) GenerateResult {
		uint8_t tokenIds[4];
		LlmRuntime state;
		uint8_t timings[112];
	};
	
	struct alignas(4) CacheBlockOptions {
		uint8_t depth[12];
	};
	
	using GpuWeightFormat = uint8_t;
	constexpr GpuWeightFormat GpuWeightFormat_f16 = 0;
	constexpr GpuWeightFormat GpuWeightFormat_f32 = 1;
	constexpr GpuWeightFormat GpuWeightFormat_wq4 = 2;
	
	using Lfm2LayerKind = uint8_t;
	constexpr Lfm2LayerKind Lfm2LayerKind_conv = 0;
	constexpr Lfm2LayerKind Lfm2LayerKind_attention = 1;
	
	struct alignas(4) Lfm2RuntimeConfig {
		double contextLength;
		double hiddenSize;
		double feedForwardSize;
		double attentionHeads;
		uint8_t kvHeadsByLayer[4];
		double headDim;
		double ropeTheta;
		double vocabSize;
		double convCacheLength;
		double normEpsilon;
		double eosToken;
		double blockCount;
		uint8_t layers[4];
		uint8_t attentionLayerSlots[4];
	};
	
	struct alignas(4) MatmulDispatchArgs {
		double rowCount;
		double tokenCount;
		double inputDim;
		double outputDim;
	};
	
	struct alignas(4) GpuTensorPage {
		GpuBuffer buffer;
		double rowStart;
		double rowCount;
		double byteLength;
	};
	
	struct alignas(4) GpuTensor {
		uint8_t name[4];
		GpuWeightFormat format;
		uint8_t _pad_format[3];
		uint8_t dimensions[4];
		uint8_t pages[4];
		uint8_t byteLength[12];
	};
	
	struct alignas(4) Lfm2RuntimeOptions {
		uint8_t contextCapacity[12];
		uint8_t maxNewTokens[12];
		uint8_t matmulKernels[64];
	};
	
	}
