/**
 * Headless, symbolic spatial perception for Pira-owned senses.
 *
 * This is deliberately not a renderer or a geometric FOV implementation. It
 * turns positions in a world snapshot into a bounded set of records which can
 * be handed to an agent. Vision has a front/left/right cone; hearing is
 * omnidirectional and therefore also exposes behind.
 */

export const SPATIAL_DISTANCE_BANDS = ["here", "near", "far"] as const;
export type SpatialDistanceBand = (typeof SPATIAL_DISTANCE_BANDS)[number];

export const SPATIAL_ANGLE_BANDS = ["front", "left", "right", "behind"] as const;
export type SpatialAngleBand = (typeof SPATIAL_ANGLE_BANDS)[number];

export const SPATIAL_MODALITIES = ["sight", "hearing"] as const;
export type SpatialSenseModality = (typeof SPATIAL_MODALITIES)[number];

export interface SpatialEntity {
	id: string;
	resourceId?: string;
	name?: string;
	state?: Record<string, unknown>;
}

export interface SpatialSenseRecord {
	kind: SpatialSenseModality;
	subject: string;
	target?: string;
	payload: {
		distance: SpatialDistanceBand;
		angle: SpatialAngleBand;
		tokens: string[];
		count: number;
		references?: string[];
	};
}

export interface SpatialSenseOptions {
	/** Modalities to emit. Defaults to both sight and hearing. */
	modalities?: readonly SpatialSenseModality[];
	/** World-unit thresholds. HERE is inclusive, NEAR is up to nearRadius. */
	hereRadius?: number;
	nearRadius?: number;
	/** Maximum number of aggregate records. Omitted means unlimited. */
	maxRecords?: number;
	/** Entity heading in radians. Zero points along +X; positive angles turn right in screen coordinates. */
	heading?: (entity: SpatialEntity) => number;
	/** Stable semantic token for the entity type, e.g. `Apple`. */
	entityToken?: (entity: SpatialEntity) => string;
	/** Additional observable tokens, e.g. `Color.Red` or `Shape.Round`. The
	 *  context says WHICH SENSE is asking and HOW FAR the thing is — a rig-aware
	 *  provider returns only the fields that sense reads at that range (and
	 *  coarser far-away token sets make far-away things GROUP). Ignoring the
	 *  second argument keeps the legacy every-field behavior. */
	observableTokens?: (entity: SpatialEntity, context: { modality: SpatialSenseModality; distance: SpatialDistanceBand }) => readonly string[];
}

const DISTANCE_ORDER: Record<SpatialDistanceBand, number> = { here: 0, near: 1, far: 2 };
const ANGLE_ORDER: Record<SpatialAngleBand, number> = { front: 0, left: 1, right: 2, behind: 3 };

/** Map a group size to the canonical Amount vocabulary used by the game. */
export function spatialAmountForCount(count: number): "littlefew" | "some" | "muchmany" {
	if (count <= 1) return "littlefew";
	if (count <= 3) return "some";
	return "muchmany";
}

function numberOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function position(entity: SpatialEntity): { x: number; y: number } | null {
	const x = numberOr(entity.state?.x, Number.NaN);
	const y = numberOr(entity.state?.y, Number.NaN);
	return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

function normalizeAngle(angle: number): number {
	let value = angle;
	while (value <= -Math.PI) value += Math.PI * 2;
	while (value > Math.PI) value -= Math.PI * 2;
	return value;
}

function distanceBand(distance: number, hereRadius: number, nearRadius: number): SpatialDistanceBand {
	if (distance <= hereRadius) return "here";
	if (distance <= nearRadius) return "near";
	return "far";
}

/** Positive relative angles are the right side in the usual screen-space XY plane. */
function angleBand(relative: number): SpatialAngleBand {
	const absolute = Math.abs(relative);
	if (absolute <= Math.PI / 4) return "front";
	if (absolute > (Math.PI * 3) / 4) return "behind";
	return relative < 0 ? "left" : "right";
}

function defaultEntityToken(entity: SpatialEntity): string {
	return String(entity.name || entity.resourceId || entity.id).trim();
}

/** Turn an entity's observable state fields into tokens, optionally limited to
 *  an allow-set of state keys (a sense's declared fields). `null` = every field
 *  — the legacy omniscient default. */
export function observableFieldTokens(entity: SpatialEntity, allowed?: ReadonlySet<string> | null): string[] {
	const tokens: string[] = [];
	for (const [key, value] of Object.entries(entity.state ?? {})) {
		if (key === "x" || key === "y" || key === "rotation") continue;
		if (key.startsWith("pending") || key.startsWith("nursery")) continue;
		if (allowed && !allowed.has(key)) continue;
		const prefix = key ? key[0]!.toUpperCase() + key.slice(1) : key;
		if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
			tokens.push(`${prefix}.${String(value)}`);
		} else if (Array.isArray(value)) {
			for (const item of value) {
				if (typeof item === "string" || typeof item === "number" || typeof item === "boolean")
					tokens.push(`${prefix}.${String(item)}`);
			}
		}
	}
	return tokens;
}

function defaultObservableTokens(entity: SpatialEntity, _context?: { modality: SpatialSenseModality; distance: SpatialDistanceBand }): string[] {
	return observableFieldTokens(entity, null); // legacy default: omniscient, range-blind
}

function uniqueTokens(tokens: readonly string[]): string[] {
	return [...new Set(tokens.map((token) => String(token).trim()).filter(Boolean))];
}

function compareStable(a: string, b: string): number {
	return a < b ? -1 : a > b ? 1 : 0;
}

interface Group {
	kind: SpatialSenseModality;
	subject: string;
	distance: SpatialDistanceBand;
	angle: SpatialAngleBand;
	tokens: string[];
	ids: string[];
}

/**
 * Share a record budget between the modalities instead of letting the first one
 * eat it.
 *
 * Records sort modality-first, so a plain head-slice hands the whole budget to
 * whichever modality comes first: a few more named objects near the subject and
 * sight fills the cap alone, hearing reports nothing, and ANG_BEHIND — which
 * ONLY hearing can produce, because vision is a cone — vanishes from the tape
 * with nothing anywhere raising an error. That is not a hypothetical: it is the
 * failure mode a JSON-key-order bug was able to hide behind.
 *
 * So: deal one record to each modality in turn, each modality drawing in its own
 * closeness order, until the budget runs out. A modality that has fewer records
 * than its share simply stops, and the rest is redealt to the others — the cap
 * is never left unspent. The RESULT keeps the canonical order (this only decides
 * WHICH records survive, never how they are laid out), so a reader's parse is
 * unaffected.
 */
export function allocateRecordBudget<T extends { kind: SpatialSenseModality }>(
	sorted: readonly T[],
	modalities: readonly SpatialSenseModality[],
	maxRecords?: number,
): T[] {
	if (maxRecords === undefined || sorted.length <= maxRecords) return [...sorted];
	if (maxRecords <= 0) return [];
	const queues = modalities.map((modality) => sorted.filter((record) => record.kind === modality));
	const cursors = queues.map(() => 0);
	const kept = new Set<T>();
	while (kept.size < maxRecords) {
		let dealt = false;
		for (let index = 0; index < queues.length && kept.size < maxRecords; index++) {
			const queue = queues[index]!;
			if (cursors[index]! >= queue.length) continue;
			kept.add(queue[cursors[index]!]!);
			cursors[index]!++;
			dealt = true;
		}
		if (!dealt) break; // every modality exhausted before the budget was
	}
	return sorted.filter((record) => kept.has(record));
}

/**
 * Build deterministic, compressed spatial sense records for one subject.
 * Entities without numeric x/y are ignored; this keeps nursery and other
 * non-spatial worlds compatible with the same provider.
 */
export function buildSpatialSenseRecords(
	subject: SpatialEntity,
	entities: Iterable<SpatialEntity>,
	options: SpatialSenseOptions = {},
): SpatialSenseRecord[] {
	const hereRadius = numberOr(options.hereRadius, 1);
	const nearRadius = numberOr(options.nearRadius, 10);
	if (hereRadius < 0 || nearRadius < hereRadius) throw new Error("nearRadius must be >= hereRadius >= 0");
	if (options.maxRecords !== undefined && (!Number.isInteger(options.maxRecords) || options.maxRecords < 0))
		throw new Error("maxRecords must be a non-negative integer");

	const subjectPosition = position(subject);
	if (!subjectPosition) return [];
	const heading = numberOr(options.heading?.(subject) ?? subject.state?.rotation, 0);
	const modalities = [...new Set(options.modalities ?? SPATIAL_MODALITIES)];
	const groups = new Map<string, Group>();

	for (const entity of entities) {
		if (!entity || entity.id === subject.id) continue;
		const targetPosition = position(entity);
		if (!targetPosition) continue;
		const dx = targetPosition.x - subjectPosition.x;
		const dy = targetPosition.y - subjectPosition.y;
		const distance = Math.hypot(dx, dy);
		const distanceClass = distanceBand(distance, hereRadius, nearRadius);
		const angle = angleBand(normalizeAngle(Math.atan2(dy, dx) - heading));
		const name = (options.entityToken ?? defaultEntityToken)(entity);

		for (const kind of modalities) {
			// Vision is a cone. Hearing is deliberately the only modality which
			// produces a BEHIND record in this first spatial vocabulary.
			if (kind === "sight" && angle === "behind") continue;
			// Tokens are PER SENSE and PER RANGE: what this modality reads of the
			// entity at this distance. Identical token sets in the same band merge
			// into one group — a coarser far-away set is what makes crowds group.
			const baseTokens = uniqueTokens([
				name,
				...(options.observableTokens ?? defaultObservableTokens)(entity, { modality: kind, distance: distanceClass }),
			]);
			const key = `${kind}|${distanceClass}|${angle}|${baseTokens.join("\u001f")}`;
			const group = groups.get(key) ?? {
				kind,
				subject: subject.id,
				distance: distanceClass,
				angle,
				tokens: baseTokens,
				ids: [],
			};
			group.ids.push(entity.id);
			groups.set(key, group);
		}
	}

	const ordered = [...groups.values()].sort((a, b) =>
		(Number(modalities.indexOf(a.kind)) - Number(modalities.indexOf(b.kind))) ||
		(DISTANCE_ORDER[a.distance] - DISTANCE_ORDER[b.distance]) ||
		(ANGLE_ORDER[a.angle] - ANGLE_ORDER[b.angle]) ||
		compareStable(a.tokens.join("\u001f"), b.tokens.join("\u001f")) ||
		compareStable(a.ids[0]!, b.ids[0]!),
	);
	// Fair share, not first-come: see allocateRecordBudget.
	const records = allocateRecordBudget(ordered, modalities, options.maxRecords)
		.map((group): SpatialSenseRecord => {
			const countToken = `Amount.${spatialAmountForCount(group.ids.length)}`;
			const record: SpatialSenseRecord = {
				kind: group.kind,
				subject: group.subject,
				payload: {
					distance: group.distance,
					angle: group.angle,
					tokens: [
						group.kind === "sight" ? "SIGHT" : "HEARING",
						`DIST_${group.distance.toUpperCase()}`,
						`ANG_${group.angle.toUpperCase()}`,
						...group.tokens,
						countToken,
					],
					count: group.ids.length,
				},
			};
			if (group.ids.length === 1) {
				record.target = group.ids[0];
				record.payload.references = [group.ids[0]!];
			}
			return record;
		});

	return records;
}
