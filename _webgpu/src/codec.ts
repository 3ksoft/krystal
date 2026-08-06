// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

import { $ } from "../../schema/src/schema";

export const {
	Lfm2Mode,
	OpParams,
	LlmRuntime,
	DecodeTelemetryEntry,
} = $.export();

export type Lfm2Mode = typeof Lfm2Mode.infer;
export type OpParams = typeof OpParams.infer;
export type LlmRuntime = typeof LlmRuntime.infer;
export type DecodeTelemetryEntry = typeof DecodeTelemetryEntry.infer;
export const SIZEOF_OpParams = 256;
export const OP_PARAMS_RESERVED_LEN = 48;
export const SIZEOF_LlmRuntime = 48;
export const SIZEOF_DecodeTelemetryEntry = 4;

export function deserializeLfm2Mode(view: DataView, offset: number): Lfm2Mode {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "prefill";
		case 1: return "decode";
		case 2: return "continuation";
		default: throw new Error("Unknown Enum value for Lfm2Mode: " + v);
	}
}

export function serializeLfm2Mode(val: Lfm2Mode, view: DataView, offset: number): void {
	if(val === "prefill") { view.setUint8(offset, 0); return; }
	if(val === "decode") { view.setUint8(offset, 1); return; }
	if(val === "continuation") { view.setUint8(offset, 2); return; }
}

export function patchLfm2Mode(path: (string | number)[], pathIdx: number, val: any, view: DataView, offset: number): void {
	serializeLfm2Mode(val, view, offset)
}

export function deserializeOpParams(view: DataView, offset: number, outObj?: any): OpParams {
	if (!outObj) {
		return {
			inputOffset: view.getUint32(offset + 0, true),
			outputOffset: view.getUint32(offset + 4, true),
			auxOffset: view.getUint32(offset + 8, true),
			aux2Offset: view.getUint32(offset + 12, true),
			tokenCount: view.getUint32(offset + 16, true),
			inputDim: view.getUint32(offset + 20, true),
			outputDim: view.getUint32(offset + 24, true),
			rowStart: view.getUint32(offset + 28, true),
			rowCount: view.getUint32(offset + 32, true),
			layerIndex: view.getUint32(offset + 36, true),
			attentionSlot: view.getUint32(offset + 40, true),
			mode: deserializeLfm2Mode(view, offset + 44),
			f0: view.getFloat32(offset + 48, true),
			f1: view.getFloat32(offset + 52, true),
			u0: view.getUint32(offset + 56, true),
			u1: view.getUint32(offset + 60, true),
			reserved: ((o) => { const a: any[] = []; for(let i=0; i<48; i++) a.push(view.getUint32(o + (i * 4), true)); return a; })(offset + 64),
		} as any;
	}
	outObj.inputOffset = view.getUint32(offset + 0, true);
	outObj.outputOffset = view.getUint32(offset + 4, true);
	outObj.auxOffset = view.getUint32(offset + 8, true);
	outObj.aux2Offset = view.getUint32(offset + 12, true);
	outObj.tokenCount = view.getUint32(offset + 16, true);
	outObj.inputDim = view.getUint32(offset + 20, true);
	outObj.outputDim = view.getUint32(offset + 24, true);
	outObj.rowStart = view.getUint32(offset + 28, true);
	outObj.rowCount = view.getUint32(offset + 32, true);
	outObj.layerIndex = view.getUint32(offset + 36, true);
	outObj.attentionSlot = view.getUint32(offset + 40, true);
	outObj.mode = deserializeLfm2Mode(view, offset + 44);
	outObj.f0 = view.getFloat32(offset + 48, true);
	outObj.f1 = view.getFloat32(offset + 52, true);
	outObj.u0 = view.getUint32(offset + 56, true);
	outObj.u1 = view.getUint32(offset + 60, true);
	outObj.reserved = ((o) => { const a: any[] = []; for(let i=0; i<48; i++) a.push(view.getUint32(o + (i * 4), true)); return a; })(offset + 64);
	return outObj;
}

export function serializeOpParams(val: OpParams, view: DataView, offset: number): void {
	view.setUint32(offset + 0, val.inputOffset, true);
	view.setUint32(offset + 4, val.outputOffset, true);
	view.setUint32(offset + 8, val.auxOffset, true);
	view.setUint32(offset + 12, val.aux2Offset, true);
	view.setUint32(offset + 16, val.tokenCount, true);
	view.setUint32(offset + 20, val.inputDim, true);
	view.setUint32(offset + 24, val.outputDim, true);
	view.setUint32(offset + 28, val.rowStart, true);
	view.setUint32(offset + 32, val.rowCount, true);
	view.setUint32(offset + 36, val.layerIndex, true);
	view.setUint32(offset + 40, val.attentionSlot, true);
	serializeLfm2Mode(val.mode, view, offset + 44);
	view.setFloat32(offset + 48, val.f0, true);
	view.setFloat32(offset + 52, val.f1, true);
	view.setUint32(offset + 56, val.u0, true);
	view.setUint32(offset + 60, val.u1, true);
	{ const o = offset + 64; for(let i=0; i<48; i++) { view.setUint32(o + (i * 4), val.reserved[i]!, true); } }
}

export function patchOpParams(path: (string | number)[], pathIdx: number, val: any, view: DataView, offset: number): void {
	if (pathIdx >= path.length) {
		serializeOpParams(val, view, offset);
		return;
	}
	const key = path[pathIdx];
	switch(key) {
		case "inputOffset":
			view.setUint32(offset + 0, val, true);
			break;
		case "outputOffset":
			view.setUint32(offset + 4, val, true);
			break;
		case "auxOffset":
			view.setUint32(offset + 8, val, true);
			break;
		case "aux2Offset":
			view.setUint32(offset + 12, val, true);
			break;
		case "tokenCount":
			view.setUint32(offset + 16, val, true);
			break;
		case "inputDim":
			view.setUint32(offset + 20, val, true);
			break;
		case "outputDim":
			view.setUint32(offset + 24, val, true);
			break;
		case "rowStart":
			view.setUint32(offset + 28, val, true);
			break;
		case "rowCount":
			view.setUint32(offset + 32, val, true);
			break;
		case "layerIndex":
			view.setUint32(offset + 36, val, true);
			break;
		case "attentionSlot":
			view.setUint32(offset + 40, val, true);
			break;
		case "mode":
			patchLfm2Mode(path, pathIdx + 1, val, view, offset + 44);
			break;
		case "f0":
			view.setFloat32(offset + 48, val, true);
			break;
		case "f1":
			view.setFloat32(offset + 52, val, true);
			break;
		case "u0":
			view.setUint32(offset + 56, val, true);
			break;
		case "u1":
			view.setUint32(offset + 60, val, true);
			break;
		case "reserved":
			if (pathIdx + 1 >= path.length) { { const o = offset + 64; for(let i=0; i<48; i++) { view.setUint32(o + (i * 4), val[i]!, true); } } return; } { const idx = path[pathIdx + 1] as number; view.setUint32(offset + 64 + (idx * 4), val, true); }
			break;
		default: break;
	}
}

export function deserializeLlmRuntimeStatus(view: DataView, offset: number): LlmRuntimeStatus {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "idle";
		case 1: return "running";
		case 2: return "eos";
		case 3: return "done";
		case 4: return "error";
		default: throw new Error("Unknown Enum value for LlmRuntimeStatus: " + v);
	}
}

export function serializeLlmRuntimeStatus(val: LlmRuntimeStatus, view: DataView, offset: number): void {
	if(val === "idle") { view.setUint8(offset, 0); return; }
	if(val === "running") { view.setUint8(offset, 1); return; }
	if(val === "eos") { view.setUint8(offset, 2); return; }
	if(val === "done") { view.setUint8(offset, 3); return; }
	if(val === "error") { view.setUint8(offset, 4); return; }
}

export function patchLlmRuntimeStatus(path: (string | number)[], pathIdx: number, val: any, view: DataView, offset: number): void {
	serializeLlmRuntimeStatus(val, view, offset)
}

export function deserializeLlmRuntime(view: DataView, offset: number, outObj?: any): LlmRuntime {
	if (!outObj) {
		return {
			contextCapacity: view.getUint32(offset + 0, true),
			maxNewTokens: view.getUint32(offset + 4, true),
			eosToken: view.getUint32(offset + 8, true),
			promptTokenCount: view.getUint32(offset + 12, true),
			position: view.getUint32(offset + 16, true),
			generatedCount: view.getUint32(offset + 20, true),
			currentToken: view.getUint32(offset + 24, true),
			status: deserializeLlmRuntimeStatus(view, offset + 28),
			telemetryRevision: view.getUint32(offset + 32, true),
			lastToken: view.getUint32(offset + 36, true),
			errorCode: view.getUint32(offset + 40, true),
			pad0: view.getUint32(offset + 44, true),
		} as any;
	}
	outObj.contextCapacity = view.getUint32(offset + 0, true);
	outObj.maxNewTokens = view.getUint32(offset + 4, true);
	outObj.eosToken = view.getUint32(offset + 8, true);
	outObj.promptTokenCount = view.getUint32(offset + 12, true);
	outObj.position = view.getUint32(offset + 16, true);
	outObj.generatedCount = view.getUint32(offset + 20, true);
	outObj.currentToken = view.getUint32(offset + 24, true);
	outObj.status = deserializeLlmRuntimeStatus(view, offset + 28);
	outObj.telemetryRevision = view.getUint32(offset + 32, true);
	outObj.lastToken = view.getUint32(offset + 36, true);
	outObj.errorCode = view.getUint32(offset + 40, true);
	outObj.pad0 = view.getUint32(offset + 44, true);
	return outObj;
}

export function serializeLlmRuntime(val: LlmRuntime, view: DataView, offset: number): void {
	view.setUint32(offset + 0, val.contextCapacity, true);
	view.setUint32(offset + 4, val.maxNewTokens, true);
	view.setUint32(offset + 8, val.eosToken, true);
	view.setUint32(offset + 12, val.promptTokenCount, true);
	view.setUint32(offset + 16, val.position, true);
	view.setUint32(offset + 20, val.generatedCount, true);
	view.setUint32(offset + 24, val.currentToken, true);
	serializeLlmRuntimeStatus(val.status, view, offset + 28);
	view.setUint32(offset + 32, val.telemetryRevision, true);
	view.setUint32(offset + 36, val.lastToken, true);
	view.setUint32(offset + 40, val.errorCode, true);
	view.setUint32(offset + 44, val.pad0, true);
}

export function patchLlmRuntime(path: (string | number)[], pathIdx: number, val: any, view: DataView, offset: number): void {
	if (pathIdx >= path.length) {
		serializeLlmRuntime(val, view, offset);
		return;
	}
	const key = path[pathIdx];
	switch(key) {
		case "contextCapacity":
			view.setUint32(offset + 0, val, true);
			break;
		case "maxNewTokens":
			view.setUint32(offset + 4, val, true);
			break;
		case "eosToken":
			view.setUint32(offset + 8, val, true);
			break;
		case "promptTokenCount":
			view.setUint32(offset + 12, val, true);
			break;
		case "position":
			view.setUint32(offset + 16, val, true);
			break;
		case "generatedCount":
			view.setUint32(offset + 20, val, true);
			break;
		case "currentToken":
			view.setUint32(offset + 24, val, true);
			break;
		case "status":
			patchLlmRuntimeStatus(path, pathIdx + 1, val, view, offset + 28);
			break;
		case "telemetryRevision":
			view.setUint32(offset + 32, val, true);
			break;
		case "lastToken":
			view.setUint32(offset + 36, val, true);
			break;
		case "errorCode":
			view.setUint32(offset + 40, val, true);
			break;
		case "pad0":
			view.setUint32(offset + 44, val, true);
			break;
		default: break;
	}
}

export function deserializeDecodeTelemetryEntry(view: DataView, offset: number, outObj?: any): DecodeTelemetryEntry {
	if (!outObj) {
		return {
			position: view.getUint8(offset + 0),
			status: (view.getUint8(offset + 1) >> 0) & 15,
			tokenId: view.getUint16(offset + 2, true),
		} as any;
	}
	outObj.position = view.getUint8(offset + 0);
	outObj.status = (view.getUint8(offset + 1) >> 0) & 15;
	outObj.tokenId = view.getUint16(offset + 2, true);
	return outObj;
}

export function serializeDecodeTelemetryEntry(val: DecodeTelemetryEntry, view: DataView, offset: number): void {
	view.setUint8(offset + 0, val.position);
	{ let _b1 = 0; _b1 |= ((val.status & 15) << 0); view.setUint8(offset + 1, _b1); }
	view.setUint16(offset + 2, val.tokenId, true);
}

export function patchDecodeTelemetryEntry(path: (string | number)[], pathIdx: number, val: any, view: DataView, offset: number): void {
	if (pathIdx >= path.length) {
		serializeDecodeTelemetryEntry(val, view, offset);
		return;
	}
	const key = path[pathIdx];
	switch(key) {
		case "position":
			view.setUint8(offset + 0, val);
			break;
		case "status":
			{ let temp = view.getUint8(offset + 1); temp &= ~(15 << 0); temp |= ((val & 15) << 0); view.setUint8(offset + 1, temp); }
			break;
		case "tokenId":
			view.setUint16(offset + 2, val, true);
			break;
		default: break;
	}
}

