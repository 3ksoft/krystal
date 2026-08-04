// generate_dataset.ts
//
// bun run generate_dataset.ts [count]
//
// Pipeline:
//   promptgen -> task
//   task -> JSON Schema
//   JSON Schema -> ArkType
//   canonical ArkType JSON Schema -> constrained value
//   schema + value -> natural language sampleText
//   SchemaAnalyzer gate
//   JSONL + stats

import { appendFile, mkdir, writeFile } from "node:fs/promises"
import { complete } from "../../../../laboratorium/dyfuzor/src/ai/llama"
import { expandByName } from "../../../../laboratorium/dyfuzor/src/ai/promptgen"
import { scope } from "arktype"
import { SchemaAnalyzer } from "@schema-pop/core"
import { jsonSchemaToType } from "@ark/json-schema"
import { exportPlan } from "@schema-pop/exporter"

const COUNT = Number(process.argv[2] ?? 100)
const OUTPUT = "./out/schema-dataset.jsonl"

const SCHEMA_ENVELOPE = {
	type: "object",
	required: ["schema"],
	properties: {
		schema: { type: "object" },
	},
} as const

const MESSAGE_ENVELOPE = {
	type: "object",
	required: ["message"],
	properties: {
		message: { type: "string" },
	},
} as const

type JsonSchema = Record<string, any>

type Stats = {
	attempts: number
	accepted: number
	rejected: {
		schema: number
		value: number
		text: number
		plan: number
		other: number
	}
	valueRetries: number
	root: Record<string, number>
	features: {
		objects: number
		arrays: number
		strings: number
		numbers: number
		integers: number
		booleans: number
		enums: number
		patterns: number
		formats: number
		optionals: number
	}
	optionalValues: {
		declared: number
		present: number
		absent: number
	}
	maxDepth: number
	totalProperties: number
}

const stats: Stats = {
	attempts: 0,
	accepted: 0,
	rejected: {
		schema: 0,
		value: 0,
		text: 0,
		plan: 0,
		other: 0,
	},
	valueRetries: 0,
	root: {},
	features: {
		objects: 0,
		arrays: 0,
		strings: 0,
		numbers: 0,
		integers: 0,
		booleans: 0,
		enums: 0,
		patterns: 0,
		formats: 0,
		optionals: 0,
	},
	optionalValues: {
		declared: 0,
		present: 0,
		absent: 0,
	},
	maxDepth: 0,
	totalProperties: 0,
}

function collectSchemaStats(
	schema: JsonSchema,
	value: unknown,
	depth = 0,
	isRoot = true,
) {
	if (!schema || typeof schema !== "object") return

	stats.maxDepth = Math.max(stats.maxDepth, depth)

	const schemaType = schema.type

	if (isRoot && typeof schemaType === "string") {
		stats.root[schemaType] = (stats.root[schemaType] ?? 0) + 1
	}

	if (Array.isArray(schema.enum)) stats.features.enums++
	if (schema.pattern !== undefined) stats.features.patterns++
	if (schema.format !== undefined) stats.features.formats++

	switch (schemaType) {
		case "object": {
			stats.features.objects++

			const properties = schema.properties ?? {}
			const required = new Set<string>(schema.required ?? [])
			const objectValue =
				value !== null && typeof value === "object" && !Array.isArray(value)
					? value as Record<string, unknown>
					: undefined

			stats.totalProperties += Object.keys(properties).length

			for (const [name, field] of Object.entries(properties)) {
				const isOptional = !required.has(name)
				const childValue = objectValue?.[name]

				if (isOptional) {
					stats.features.optionals++
					stats.optionalValues.declared++

					if (objectValue && Object.prototype.hasOwnProperty.call(objectValue, name)) {
						stats.optionalValues.present++
					} else {
						stats.optionalValues.absent++
					}
				}

				collectSchemaStats(field as JsonSchema, childValue, depth + 1, false)
			}

			break
		}

		case "array": {
			stats.features.arrays++

			const items = Array.isArray(value) ? value : []

			if (schema.items) {
				if (items.length === 0) {
					collectSchemaStats(schema.items, undefined, depth + 1, false)
				} else {
					collectSchemaStats(schema.items, items[0], depth + 1, false)

					for (let i = 1; i < items.length; i++) {
						collectOptionalPresenceOnly(schema.items, items[i])
					}
				}
			}

			break
		}

		case "string":
			stats.features.strings++
			break
		case "number":
			stats.features.numbers++
			break
		case "integer":
			stats.features.integers++
			break
		case "boolean":
			stats.features.booleans++
			break
	}

	for (const key of ["anyOf", "oneOf", "allOf"]) {
		if (Array.isArray(schema[key])) {
			for (const variant of schema[key]) {
				collectSchemaStats(variant, value, depth + 1, false)
			}
		}
	}
}

function collectOptionalPresenceOnly(schema: JsonSchema, value: unknown) {
	if (!schema || typeof schema !== "object") return

	if (schema.type === "object") {
		const properties = schema.properties ?? {}
		const required = new Set<string>(schema.required ?? [])
		const objectValue =
			value !== null && typeof value === "object" && !Array.isArray(value)
				? value as Record<string, unknown>
				: undefined

		for (const [name, field] of Object.entries(properties)) {
			if (!required.has(name)) {
				stats.optionalValues.declared++

				if (objectValue && Object.prototype.hasOwnProperty.call(objectValue, name)) {
					stats.optionalValues.present++
				} else {
					stats.optionalValues.absent++
				}
			}

			collectOptionalPresenceOnly(field as JsonSchema, objectValue?.[name])
		}

		return
	}

	if (schema.type === "array" && schema.items && Array.isArray(value)) {
		for (const item of value) {
			collectOptionalPresenceOnly(schema.items, item)
		}
	}

	for (const key of ["anyOf", "oneOf", "allOf"]) {
		if (Array.isArray(schema[key])) {
			for (const variant of schema[key]) {
				collectOptionalPresenceOnly(variant, value)
			}
		}
	}
}

function printStats() {
	const acceptanceRate =
		stats.attempts > 0
			? ((stats.accepted / stats.attempts) * 100).toFixed(1)
			: "0.0"

	const optionalPresenceRate =
		stats.optionalValues.declared > 0
			? ((stats.optionalValues.present / stats.optionalValues.declared) * 100).toFixed(1)
			: "n/a"

	console.log("\n--- dataset stats ---")
	console.log(`attempts: ${stats.attempts}`)
	console.log(`accepted: ${stats.accepted} (${acceptanceRate}%)`)
	console.log(`value retries: ${stats.valueRetries}`)
	console.log(`optional present: ${optionalPresenceRate}%`)
	console.log(JSON.stringify(stats, null, 2))
	console.log("---------------------\n")
}

async function generateSchema(text: string, seed: number) {
	const result = await complete({
		model: "qwen3.5-4b",
		schema: SCHEMA_ENVELOPE,
		temperature: 0.4,
		maxTokens: 1024,
		seed,
		prompt: `
Extract the parameters from the following user request and return a JSON schema that would describe the request.

Identify a subset of the parameters, choose most essential ones. Include those in appriopriate 'required' section in the schema.

RECURSIVE COMPLETENESS: Every object field (including nested objects like 'metadata' or 'root') MUST explicitly define its internal 'properties' down to primitive types (number, string, boolean, enum). NEVER output a plain {"type": "object"} without an internal "properties" block.

SEMANTIC PROPERTY NAMES: Format guidelines (like 'email', 'uuid', 'date-time') describe the TYPE format of a field, NOT its property name. Use domain-relevant field names (e.g. 'contact_email', 'device_uuid', 'recorded_at') instead of literal format strings.

NO TYPES OF UNKNOWN LENGTH: Provide maxLength for each string value and maxItems for every array.

DO NOT RESPOND TO THE FOLLOWING REQUEST, ONLY RETURN THE JSON SCHEMA THAT DESCRIBES THE REQUEST.

----
${text}
----
		`.trim(),
	})

	const json = result.json as { schema?: JsonSchema }

	if (!json?.schema || typeof json.schema !== "object") {
		throw new Error("invalid schema")
	}

	return json.schema
}

async function generateText(
	schema: JsonSchema,
	values: unknown,
	seed: number,
): Promise<string> {
	const result = await complete({
		model: "qwen3.5-4b",
		temperature: 0.2,
		maxTokens: 256,
		schema: MESSAGE_ENVELOPE,
		seed,
		prompt: `

------

VALUE:
${JSON.stringify(values, null, 2)}

SCHEMA:
${JSON.stringify(schema, null, 2)}

-----

Create a natural user message which would provided above.

Include ALL facts provided.

Do not invent facts that contradict the provided value.

Express the information naturally rather than mechanically enumerating fields.

Use a simple conversational style.

		`.trim(),
	})

	const message = (result.json as { message?: unknown } | undefined)?.message

	if (typeof message !== "string" || message.length === 0) {
		throw new Error("Could not generate sample text")
	}

	return message
}

async function generateSample(
	schema: JsonSchema,
	task: string,
	seed: number,
): Promise<unknown> {
	const result = await complete({
		model: "qwen3.5-4b",
		schema,
		temperature: 0.3,
		maxTokens: 256,
		seed,
		prompt: `Generate sample plausible data set for ${task}, Your response must adhere to the following schema: ${JSON.stringify(schema, null, 2)}`,
	})

	return result.json
}

await mkdir("./out", { recursive: true })
await writeFile(OUTPUT, "")

let done = 0
let n = 103

while (done < COUNT) {
	stats.attempts++

	let stage: "schema" | "value" | "text" | "plan" | "other" = "schema"

	try {
		const task = await expandByName("schema", n )
		const schema_shape = await expandByName("schema-shape", n)

		stage = "schema"
		const rawSchema = await generateSchema(task + " " + schema_shape, n)

		const arkType = jsonSchemaToType(rawSchema)

		if (arkType === undefined) {
			throw new Error("Could not convert schema to ArkType")
		}


		const module = scope({ value: arkType })
		const analyzer = new SchemaAnalyzer()
		const analysis = analyzer.analyze(module, { mode : "binary"})

		if (!analysis.plan || analysis.errors.length > 0) {
			throw new Error("Could not generate plan for schema " + JSON.stringify(analysis.errors, null, 2));
		}

		
		const plan = exportPlan(analysis.plan, "outputPlan");

		const sampleSchema = arkType.toJsonSchema() as JsonSchema
		delete sampleSchema.$schema

		stage = "value"

		let sample: unknown
		let retries = 5

		console.log("Schema accepted, generating value...")

		for (;;) {
			sample = await generateSample(sampleSchema, task, n + 2 * retries)

			if (sample !== undefined && arkType.allows(sample)) {
				console.log("Sample value conforms to schema.")
				break
			}

			stats.valueRetries++
			retries--

			if (retries <= 0) {
				throw new Error(
					"Could not generate valid sample value for schema " +
						JSON.stringify(sampleSchema) +
						" last sample: " +
						JSON.stringify(sample),
				)
			}

			console.warn(
				`Sample value does not conform to schema, retrying... (${retries} retries left)`,
			)
		}

		stage = "text"
		const sampleText = await generateText(sampleSchema, sample, n)

		console.log("Generated sample text:", sampleText)

		stage = "plan"

		collectSchemaStats(sampleSchema, sample)

		await appendFile(
			OUTPUT,
			JSON.stringify({
				seed: n,
				id: done,
				task,
				schema: sampleSchema,
				value: sample,
				sampleText,
				analysis,
				plan
			}) + "\n",
		)

		done++
		stats.accepted++

		console.log(`[${done}/${COUNT}] ${sampleText.slice(0, 100)}`)

		if (done % 10 === 0 || done === COUNT) {
			printStats()
		}
	} catch (err) {
		if (stage in stats.rejected) {
			stats.rejected[stage as keyof typeof stats.rejected]++
		} else {
			stats.rejected.other++
		}

		console.warn(
			`❗ [reject:${stage}] ${err instanceof Error ? err.message : String(err)}`,
		)
	} finally {
		n++
	}
}

printStats()