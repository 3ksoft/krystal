#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <array>

namespace v1_0_0 {
	using EngineEventTag = uint8_t;
	constexpr EngineEventTag EngineEventTag_Completed = 0;
	constexpr EngineEventTag EngineEventTag_ExecutionStats = 1;
	constexpr EngineEventTag EngineEventTag_Failed = 2;
	constexpr EngineEventTag EngineEventTag_TokenEmitted = 3;

	using EngineCommandTag = uint8_t;
	constexpr EngineCommandTag EngineCommandTag_Cancel = 0;
	constexpr EngineCommandTag EngineCommandTag_CreateCheckpoint = 1;
	constexpr EngineCommandTag EngineCommandTag_DropBlock = 2;
	constexpr EngineCommandTag EngineCommandTag_DropCheckpoint = 3;
	constexpr EngineCommandTag EngineCommandTag_Generate = 4;
	constexpr EngineCommandTag EngineCommandTag_PutBlock = 5;

	using FrameDirection = uint8_t;
	constexpr FrameDirection FrameDirection_command = 0;
	constexpr FrameDirection FrameDirection_event = 1;

	struct alignas(1) FrameHeader {
		uint32_t magic;
		uint16_t version;
		FrameDirection direction;
		uint8_t flags;
		uint32_t bodyBytes;
		uint32_t payloadBytes;
	};

	struct alignas(1) ContextRef {
		uint32_t checkpoint;
		uint16_t blockCount;
		uint16_t reserved;
	};

	struct alignas(1) PutBlock {
		uint32_t operation;
		uint32_t block;
		uint32_t tokenCount;
	};

	struct alignas(1) DropBlock {
		uint32_t operation;
		uint32_t block;
	};

	struct alignas(1) CreateCheckpoint {
		uint32_t operation;
		uint32_t checkpoint;
		ContextRef context;
	};

	struct alignas(1) DropCheckpoint {
		uint32_t operation;
		uint32_t checkpoint;
	};

	using Sampler = uint8_t;
	constexpr Sampler Sampler_argmax = 0;
	constexpr Sampler Sampler_topk = 1;

	struct alignas(1) Generate {
		uint32_t operation;
		ContextRef context;
		uint32_t maxTokens;
		float temperature;
		uint32_t seed;
		uint16_t topK;
		Sampler sampler;
		uint8_t reserved;
	};

	struct alignas(1) Cancel {
		uint32_t operation;
		uint32_t target;
	};

	struct alignas(1) EngineCommand { uint8_t _bytes[29]; };

	using ErrorCode = uint8_t;
	constexpr ErrorCode ErrorCode_InvalidCommand = 0;
	constexpr ErrorCode ErrorCode_InvalidContext = 1;
	constexpr ErrorCode ErrorCode_NotFound = 2;
	constexpr ErrorCode ErrorCode_CapacityExceeded = 3;
	constexpr ErrorCode ErrorCode_Cancelled = 4;
	constexpr ErrorCode ErrorCode_InternalError = 5;

	struct alignas(1) Completed {
		uint32_t operation;
	};

	struct alignas(1) TokenEmitted {
		uint32_t operation;
		uint32_t token;
	};

	struct alignas(1) ExecutionStats {
		uint32_t operation;
		uint32_t prefillTokens;
		uint32_t checkpointHits;
		uint32_t checkpointMisses;
		uint32_t restoredBytes;
		uint32_t checkpointBytes;
		uint32_t kvBytes;
		uint32_t kvCapacityBytes;
		uint32_t convBytes;
		uint32_t hiddenBytes;
		uint32_t checkpointCreateUs;
		uint32_t checkpointRestoreUs;
	};

	struct alignas(1) Failed {
		uint32_t operation;
		uint16_t messageBytes;
		ErrorCode code;
		uint8_t reserved;
	};

	struct alignas(1) EngineEvent { uint8_t _bytes[49]; };

	}
