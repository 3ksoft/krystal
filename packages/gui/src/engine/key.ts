import type { InjectionKey } from "vue";
import type { EngineApi } from "./useEngine.ts";

export const ENGINE_KEY: InjectionKey<EngineApi> = Symbol("chomato.engine");
