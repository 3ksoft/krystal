#include <cstddef>
#include "../generated/bridge.hpp"
using namespace v1_0_0;

static_assert(sizeof(FrameHeader) == 16);
static_assert(sizeof(ContextRef) == 8);
static_assert(sizeof(PutBlock) == 12);
static_assert(sizeof(CreateCheckpoint) == 16);
static_assert(sizeof(Generate) == 28);
static_assert(sizeof(EngineCommand) == 29);
static_assert(sizeof(ExecutionStats) == 48);
static_assert(sizeof(Failed) == 8);
static_assert(sizeof(EngineEvent) == 49);

static_assert(offsetof(PutBlock, operation) == 0);
static_assert(offsetof(PutBlock, block) == 4);
static_assert(offsetof(PutBlock, tokenCount) == 8);
static_assert(offsetof(CreateCheckpoint, context) == 8);
static_assert(offsetof(Generate, context) == 4);
static_assert(offsetof(Generate, maxTokens) == 12);
// The sampling block is ordered widest-first so that this natural C++ layout
// stays byte-identical to the packed wire layout; the struct is alignas(1) but
// not packed, so a narrower field placed before `temperature` would silently
// insert padding here and nowhere else.
static_assert(offsetof(Generate, temperature) == 16);
static_assert(offsetof(Generate, seed) == 20);
static_assert(offsetof(Generate, topK) == 24);
static_assert(offsetof(Generate, sampler) == 26);
static_assert(offsetof(Generate, reserved) == 27);
static_assert(offsetof(ExecutionStats, operation) == 0);
static_assert(offsetof(ExecutionStats, prefillTokens) == 4);
static_assert(offsetof(ExecutionStats, checkpointHits) == 8);
static_assert(offsetof(ExecutionStats, checkpointMisses) == 12);
static_assert(offsetof(ExecutionStats, restoredBytes) == 16);
static_assert(offsetof(ExecutionStats, checkpointBytes) == 20);
static_assert(offsetof(ExecutionStats, kvBytes) == 24);
static_assert(offsetof(ExecutionStats, kvCapacityBytes) == 28);
static_assert(offsetof(ExecutionStats, convBytes) == 32);
static_assert(offsetof(ExecutionStats, hiddenBytes) == 36);
static_assert(offsetof(ExecutionStats, checkpointCreateUs) == 40);
static_assert(offsetof(ExecutionStats, checkpointRestoreUs) == 44);
static_assert(offsetof(Failed, operation) == 0);
static_assert(offsetof(Failed, messageBytes) == 4);
static_assert(offsetof(Failed, code) == 6);
static_assert(offsetof(Failed, reserved) == 7);

int main() { return 0; }
