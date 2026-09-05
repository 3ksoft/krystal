// THIS FILE IS AUTO-GENERATED - DO NOT CHANGE

import type { v1_0_0 } from "./krystal.types.ts";

type BrainBandKind = v1_0_0.BrainBandKind;
type BrainValueKind = v1_0_0.BrainValueKind;
type BrainFrameGpu = v1_0_0.BrainFrameGpu;

const __textDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;
const __textEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;

export const SIZEOF_BrainFrameGpu = 32832;
export const BRAIN_FRAME_GPU_TOKENIDS_LEN = 3456;
export const BRAIN_FRAME_GPU_FIELDROLES_LEN = 3456;
export const BRAIN_FRAME_GPU_SCHEMAIDS_LEN = 432;
export const BRAIN_FRAME_GPU_BANDIDS_LEN = 432;
export const BRAIN_FRAME_GPU_ACTIVERECORDINDICES_LEN = 432;

export function deserializeBrainBandKind(view: DataView, offset: number): BrainBandKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "system";
		case 1: return "homeostasis";
		case 2: return "body";
		case 3: return "perception";
		case 4: return "memory";
		case 5: return "focus";
		case 6: return "query";
		case 7: return "catalog";
		default: throw new Error("Unknown Enum value for BrainBandKind: " + v);
	}
}

export function serializeBrainBandKind(val: BrainBandKind, view: DataView, offset: number): void {
	if(val === "system") { view.setUint8(offset, 0); return; }
	if(val === "homeostasis") { view.setUint8(offset, 1); return; }
	if(val === "body") { view.setUint8(offset, 2); return; }
	if(val === "perception") { view.setUint8(offset, 3); return; }
	if(val === "memory") { view.setUint8(offset, 4); return; }
	if(val === "focus") { view.setUint8(offset, 5); return; }
	if(val === "query") { view.setUint8(offset, 6); return; }
	if(val === "catalog") { view.setUint8(offset, 7); return; }
}

export function deserializeBrainValueKind(view: DataView, offset: number): BrainValueKind {
	const v = view.getUint8(offset);
	switch(v) {
		case 0: return "none";
		case 1: return "token";
		case 2: return "context_ref";
		case 3: return "record_ref";
		case 4: return "boolean_class";
		case 5: return "scalar_band";
		case 6: return "quantity_projection";
		case 7: return "opaque_payload";
		default: throw new Error("Unknown Enum value for BrainValueKind: " + v);
	}
}

export function serializeBrainValueKind(val: BrainValueKind, view: DataView, offset: number): void {
	if(val === "none") { view.setUint8(offset, 0); return; }
	if(val === "token") { view.setUint8(offset, 1); return; }
	if(val === "context_ref") { view.setUint8(offset, 2); return; }
	if(val === "record_ref") { view.setUint8(offset, 3); return; }
	if(val === "boolean_class") { view.setUint8(offset, 4); return; }
	if(val === "scalar_band") { view.setUint8(offset, 5); return; }
	if(val === "quantity_projection") { view.setUint8(offset, 6); return; }
	if(val === "opaque_payload") { view.setUint8(offset, 7); return; }
}

export function deserializeBrainFrameGpu(view: DataView, offset: number, outObj?: any): BrainFrameGpu {
	if (!outObj) {
		const _arr_tokenIds = new Array(3456);
		for (let i = 0, _off_tokenIds = offset; i < 3456; i++, _off_tokenIds += 4) {
			_arr_tokenIds[i] = view.getUint32(_off_tokenIds, true);
		}
		const _arr_fieldRoles = new Array(3456);
		for (let i = 0, _off_fieldRoles = offset + 13824; i < 3456; i++, _off_fieldRoles += 4) {
			_arr_fieldRoles[i] = view.getUint32(_off_fieldRoles, true);
		}
		const _arr_schemaIds = new Array(432);
		for (let i = 0, _off_schemaIds = offset + 27648; i < 432; i++, _off_schemaIds += 4) {
			_arr_schemaIds[i] = view.getUint32(_off_schemaIds, true);
		}
		const _arr_bandIds = new Array(432);
		for (let i = 0, _off_bandIds = offset + 29376; i < 432; i++, _off_bandIds += 4) {
			_arr_bandIds[i] = view.getUint32(_off_bandIds, true);
		}
		const _arr_activeRecordIndices = new Array(432);
		for (let i = 0, _off_activeRecordIndices = offset + 31104; i < 432; i++, _off_activeRecordIndices += 4) {
			_arr_activeRecordIndices[i] = view.getUint32(_off_activeRecordIndices, true);
		}
		return {
			tokenIds: _arr_tokenIds,
			fieldRoles: _arr_fieldRoles,
			schemaIds: _arr_schemaIds,
			bandIds: _arr_bandIds,
			activeRecordIndices: _arr_activeRecordIndices,
		} as any;
	}
	const _arr_tokenIds = new Array(3456);
	for (let i = 0, _off_tokenIds = offset; i < 3456; i++, _off_tokenIds += 4) {
		_arr_tokenIds[i] = view.getUint32(_off_tokenIds, true);
	}
	const _arr_fieldRoles = new Array(3456);
	for (let i = 0, _off_fieldRoles = offset + 13824; i < 3456; i++, _off_fieldRoles += 4) {
		_arr_fieldRoles[i] = view.getUint32(_off_fieldRoles, true);
	}
	const _arr_schemaIds = new Array(432);
	for (let i = 0, _off_schemaIds = offset + 27648; i < 432; i++, _off_schemaIds += 4) {
		_arr_schemaIds[i] = view.getUint32(_off_schemaIds, true);
	}
	const _arr_bandIds = new Array(432);
	for (let i = 0, _off_bandIds = offset + 29376; i < 432; i++, _off_bandIds += 4) {
		_arr_bandIds[i] = view.getUint32(_off_bandIds, true);
	}
	const _arr_activeRecordIndices = new Array(432);
	for (let i = 0, _off_activeRecordIndices = offset + 31104; i < 432; i++, _off_activeRecordIndices += 4) {
		_arr_activeRecordIndices[i] = view.getUint32(_off_activeRecordIndices, true);
	}
	outObj.tokenIds = _arr_tokenIds;
	outObj.fieldRoles = _arr_fieldRoles;
	outObj.schemaIds = _arr_schemaIds;
	outObj.bandIds = _arr_bandIds;
	outObj.activeRecordIndices = _arr_activeRecordIndices;
	return outObj;
}

export function serializeBrainFrameGpu(val: BrainFrameGpu, view: DataView, offset: number): void {
	{ for (let i = 0, __o = offset; i < 3456; i++, __o += 4) { const __e = val.tokenIds[i]!; view.setUint32(__o, __e, true); } }
	{ for (let i = 0, __o = offset + 13824; i < 3456; i++, __o += 4) { const __e = val.fieldRoles[i]!; view.setUint32(__o, __e, true); } }
	{ for (let i = 0, __o = offset + 27648; i < 432; i++, __o += 4) { const __e = val.schemaIds[i]!; view.setUint32(__o, __e, true); } }
	{ for (let i = 0, __o = offset + 29376; i < 432; i++, __o += 4) { const __e = val.bandIds[i]!; view.setUint32(__o, __e, true); } }
	{ for (let i = 0, __o = offset + 31104; i < 432; i++, __o += 4) { const __e = val.activeRecordIndices[i]!; view.setUint32(__o, __e, true); } }
}

