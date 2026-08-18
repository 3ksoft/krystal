import {
  BRAIN_FIXED_RECORDS,
  BRAIN_FRAME_BANDS,
  BRAIN_LIMITS,
  RECORD_FLAGS,
  TOKEN_FLAGS,
} from "../packages/schema/src/krystal-engine-schema.ts";
import type { v1_0_0 } from "../packages/schema/generated/krystal.types.ts";
import { PAD_TOKEN_ID } from "../packages/krystal/src/frame/packer.ts";
import { fixtureTokenId } from "../packages/krystal/src/fixtures/vocabulary.ts";
import { ACTION_INTENT_SCHEMA_ID } from "../packages/krystal/src/fixtures/frame.ts";
import { mulberry32 } from "../packages/krystal/src/forward/model.ts";
import { policyRefToken } from "../packages/krystal/src/bridge/policy.ts";

export interface CaseBindExample {
  readonly seed: number;
  readonly frame: v1_0_0.BrainFrame;
  readonly wordIds: Record<number, number>;
  readonly gold: {
    readonly action: "CHASE";
    readonly refToken: number;
    readonly targetSlot: number;
  };
  readonly patientNoun: "DOG" | "CAT";
}

export interface CaseBindOptions {
  /** Wymieszaj powiązania wordIds (negatywna kontrola - broken binding). */
  readonly scrambleBinding?: boolean;
}

const VISION_BAND = BRAIN_FRAME_BANDS.find((b) => b.kind === "vision")!;

export function createCaseBindFrame(seed: number, options: CaseBindOptions = {}): CaseBindExample {
  const rng = mulberry32(seed * 31 + 7);
  const recordWidth = BRAIN_LIMITS.recordWidth;

  // 1. Losujemy role: kto jest celem (biernik / patient)?
  const isDogPatient = (rng() > 0.5);
  const patientNoun = isDogPatient ? "DOG" : "CAT";

  const dogRef = policyRefToken(seed, 0, "train");
  const catRef = policyRefToken(seed, 1, "train");
  const goldRef = isDogPatient ? dogRef : catRef;

  // 2. Losujemy sloty w pasmie vision (kolejność w ramce nie może zdradzać odpowiedzi)
  const slotA = VISION_BAND.recordOffset + 0;
  const slotB = VISION_BAND.recordOffset + 1;
  const swapSlots = rng() > 0.5;
  const dogSlot = swapSlots ? slotB : slotA;
  const catSlot = swapSlots ? slotA : slotB;
  const targetSlot = isDogPatient ? dogSlot : catSlot;
  const distractorSlot = isDogPatient ? catSlot : dogSlot;

  // 3. Inicjalizacja pustej ramki
  const records: v1_0_0.BrainRecordSlot[] = Array.from({ length: BRAIN_LIMITS.frameRecordSlots }, () => ({
    header: {
      schemaId: 0, band: "system", source: "runtime", flags: 0,
      tokenCount: 0, referenceCount: 0, observedAt: 0, revision: 0,
      primaryReference: 0xffffffff, continuationRecord: 0xffffffff,
      salience: 0, freshness: 0,
    },
    tokens: new Array<number>(recordWidth).fill(PAD_TOKEN_ID),
    tokenMeta: new Array<v1_0_0.BrainTokenMeta>(recordWidth).fill({
      fieldId: 0, roleToken: 0, flags: TOKEN_FLAGS.padding, referenceBinding: 0xffffffff,
    }),
    references: [],
  }));

  // Pomocnik do wpisania rekordu
  const writeRecord = (slot: number, schemaId: number, band: v1_0_0.BrainBandKind, tokens: number[], refToken?: number) => {
    const padded = [...tokens, ...new Array<number>(Math.max(0, recordWidth - tokens.length)).fill(PAD_TOKEN_ID)];
    records[slot] = {
      header: {
        schemaId, band, source: "sensor", flags: RECORD_FLAGS.occupied,
        tokenCount: tokens.length, referenceCount: refToken !== undefined ? 1 : 0,
        observedAt: 1, revision: 1, primaryReference: refToken ?? 0xffffffff,
        continuationRecord: 0xffffffff, salience: 1.0, freshness: 1.0,
      },
      tokens: padded,
      tokenMeta: padded.map((t) => ({
        fieldId: 0, roleToken: t === PAD_TOKEN_ID ? 0 : t,
        flags: t === PAD_TOKEN_ID ? TOKEN_FLAGS.padding : TOKEN_FLAGS.structural,
        referenceBinding: 0xffffffff,
      })),
      references: refToken !== undefined ? [{
        localTokenIndex: 1, fieldId: 0, flags: 0, reserved0: 0,
        handle: { tokenId: refToken, generation: 1, kind: "entity", status: "live" },
      }] : [],
    };
  };

  // 4. Wpisujemy obiekty (DOG i CAT) — jeden otrzymuje cechę ACCUSATIVE
  const dogProps = isDogPatient ? [fixtureTokenId("ACCUSATIVE")] : [];
  const catProps = !isDogPatient ? [fixtureTokenId("ACCUSATIVE")] : [];

  writeRecord(dogSlot, 20, "vision", [fixtureTokenId("DOG"), dogRef, ...dogProps], dogRef);
  writeRecord(catSlot, 21, "vision", [fixtureTokenId("CAT"), catRef, ...catProps], catRef);

  // 5. Query / Akcja CHASE
  const querySlot = BRAIN_FIXED_RECORDS.primaryQuery;
  const chaseActionSlot = BRAIN_FIXED_RECORDS.catalogBase;
  writeRecord(querySlot, 3, "query", [fixtureTokenId("CHASE")]);
  writeRecord(chaseActionSlot, ACTION_INTENT_SCHEMA_ID, "catalog", [fixtureTokenId("CHASE")]);

  // 6. Definicja Word Attention Bias:
  // Wiążemy wspólny identyfikator (np. wordId = 101) pomiędzy slotem intencji/query a slotem z biernikiem
  const BINDING_WORD_ID = 101;
  const wordIds: Record<number, number> = {};

  if (options.scrambleBinding) {
    // Negatywna kontrola: bias wskazuje na distractor zamiast na obiekt w bierniku
    wordIds[querySlot] = BINDING_WORD_ID;
    wordIds[chaseActionSlot] = BINDING_WORD_ID;
    wordIds[distractorSlot] = BINDING_WORD_ID;
  } else {
    // Poprawne powiązanie: predykat <-> rzeczownik w bierniku
    wordIds[querySlot] = BINDING_WORD_ID;
    wordIds[chaseActionSlot] = BINDING_WORD_ID;
    wordIds[targetSlot] = BINDING_WORD_ID;
  }

  const frame: v1_0_0.BrainFrame = {
    header: {
      tokenAbiVersion: 0, architectureVersion: 2, layoutVersion: 1, tick: 1, snapshot: 1,
      activeRecordCount: 4, activeTokenCount: 10, activeQueryRecord: querySlot,
      actorRecord: BRAIN_FIXED_RECORDS.actor, frameRevision: 1, memoryRevision: 0,
      intentRevision: 0, flags: 0,
    },
    bands: BRAIN_FRAME_BANDS.map((b) => ({
      kind: b.kind,
      activeRecords: b.kind === "vision" ? 2 : b.kind === "catalog" || b.kind === "query" ? 1 : 0,
      activeTokens: 0, overflowRecords: 0, truncatedRecords: 0, revision: 0, flags: 0, reserved0: 0,
    })),
    records,
  };

  return {
    seed,
    frame,
    wordIds,
    gold: { action: "CHASE", refToken: goldRef, targetSlot },
    patientNoun,
  };
}

export function caseBindPairs(seeds: number[], options: CaseBindOptions = {}): CaseBindExample[] {
  return seeds.map((s) => createCaseBindFrame(s, options));
}