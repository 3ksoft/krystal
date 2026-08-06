#include <cstddef>
#include "../generated/bridge.hpp"
using namespace v1_0_0;

static_assert(sizeof(FrameHeader) == 16);
static_assert(sizeof(ContextRef) == 8);
static_assert(sizeof(PutBlock) == 12);
static_assert(sizeof(CreateCheckpoint) == 16);
static_assert(sizeof(Generate) == 16);
static_assert(sizeof(EngineCommand) == 17);
static_assert(sizeof(Failed) == 8);
static_assert(sizeof(EngineEvent) == 9);

static_assert(offsetof(PutBlock, operation) == 0);
static_assert(offsetof(PutBlock, block) == 4);
static_assert(offsetof(PutBlock, tokenCount) == 8);
static_assert(offsetof(CreateCheckpoint, context) == 8);
static_assert(offsetof(Generate, context) == 4);
static_assert(offsetof(Generate, maxTokens) == 12);
static_assert(offsetof(Failed, operation) == 0);
static_assert(offsetof(Failed, messageBytes) == 4);
static_assert(offsetof(Failed, code) == 6);
static_assert(offsetof(Failed, reserved) == 7);

int main() { return 0; }
