
### Główna koncepcja API

API dzieli się na 4 główne filary:
1. **`ChomatoClient`**: Zarządzanie połączeniem, telemetryką i operacjami poziomu daemona.
2. **`BlockStore`**: Zarządzanie trwałymi blokami pamięci GPU (indeksowanie, zwalnianie).
3. **`Context` & `Branch`**: Drzewiasta struktura kontekstu, manipulacja blokami i rozgałęzianie (forking/checkpointing).
4. **`GenerationHandle`**: Kontrola nad strumieniowaniem generacji i anulowaniem (`CancelCurrent`).

---

### 1. Połączenie i Cykl Życia (`ChomatoClient`)

```typescript
import { ChomatoClient } from '@chomato/sdk';

// Łączenie z istniejącym daemonem (lub uruchamianie własnego)
await using client = await ChomatoClient.connect({
  transport: 'unix',
  path: '/tmp/chomato-a.sock',
});

// Telemetria (zgodnie z ADA-0001: nieblokująca, obserwacyjna)
client.telemetry.on('stats', (stats) => {
  console.log(`GPU VRAM: ${stats.vramUsedMB}MB | Cache Hits: ${stats.cacheHitRate}%`);
});
```

---

### 2. Zarządzanie Blokami (`BlockStore`)

Bloki to przeliczone / zapamiętane w `BlockStore` fragmenty (np. system prompty, dużepliki kodu, dokumentacja).

```typescript
// 1. Tworzenie i indeksowanie bloku
const systemBlock = await client.blocks.index({
  id: 'sys-coder-v1', // opcjonalne id
  content: 'Jesteś ekspertem TypeScript i architektury oprogramowania...',
});

// Indeksowanie dużego pliku/kodu z podglądem postępu (long-running operation)
const codebaseBlock = await client.blocks.index({
  content: largeCodebaseString,
  onProgress: (progress) => console.log(`Indeksowanie: ${progress.percent}%`),
});

// 2. Pobieranie informacji o blokach
const activeBlocks = await client.blocks.list();

// 3. Usuwanie z VRAM / Garbage Collection
await codebaseBlock.unload(); // UnloadBlock z ADA-0001
await client.blocks.gc();      // GarbageCollect
```

---

### 3. Kontekst, Rozgałęzienia (Branching) i Checkpointy

Sercem Chomato jest praca z kontekstem. Kontekst to dynamiczne drzewo złożone z **Bloków** oraz **Wiadomości/Tokenów**.

#### A. Tworzenie kontekstu bazowego

```typescript
// Tworzymy bazowy kontekst i podpinamy bloki
const baseCtx = client.createContext()
  .attachBlock(systemBlock)
  .attachBlock(codebaseBlock)
  .appendUser("Przeanalizujmy architekturę systemu.");
```

#### B. Rozgałęzianie (Forking / Branching)

Ponieważ daemon obsługuje sekwencyjne przetwarzanie, branching na poziomie klienta tworzy lekkie kopie struktury kontekstu, wywołując `CreateCheckpoint` w daemonie tylko wtedy, gdy jest to potrzebne.

```typescript
// Tworzymy dwie osobne gałęzie z tego samego punktu wyjścia
const branchA = baseCtx.fork();
const branchB = baseCtx.fork();

// W gałęzi A idziemy w stronę refaktoryzacji
branchA.appendUser("Propozycja A: Przepiszmy to na mikroserwisy.");

// W gałęzi B idziemy w stronę monolitu
branchB.appendUser("Propozycja B: Zostawmy monolit, ale dodajmy CQS.");
```

#### C. Trwałe Checkpointy (`CreateCheckpoint` / `RestoreCheckpoint`)

Dla bardziej złożonych operacji (np. cofanie stanu w IDE / LSP):

```typescript
// Zapisujemy punkt kontrolny w daemonie
const checkpoint = await baseCtx.saveCheckpoint('before-refactor');

// Później w aplikacji możemy przywrócić kontekst z checkpointu:
const restoredCtx = await client.restoreCheckpoint(checkpoint.id);
```

---

### 4. Generowanie, Strumieniowanie i Anulowanie

Generowanie zwraca obiekt `GenerationHandle`, który jest **AsyncIterable** (można go iterować za pomocą `for await`), ale ma też metody sterujące (`cancel`).

```typescript
// 1. Zwykłe strumieniowanie z auto-cancellation przez AbortController
const abortController = new AbortController();

const handle = branchA.generate({
  temperature: 0.2,
  signal: abortController.signal,
});

// Iteracja po tokenach (TokenEmitted)
for await (const token of handle) {
  process.stdout.write(token);
}

// Otrzymanie pełnych metadanych po zakończeniu
const result = await handle.result();
console.log(`Wygenerowano ${result.tokensCount} tokenów w ${result.durationMs}ms`);
```

#### Anulowanie zapytania (`CancelCurrent`)

Zgodnie z ADA-0001, anulowanie wywołuje `CancelCurrent` na daemonie.

```typescript
const handle = branchB.generate({ maxTokens: 1000 });

// Anulowanie z dowolnego miejsca w kodzie
setTimeout(async () => {
  console.log("Anulowanie żądania...");
  await handle.cancel(); // Wysyła CancelCurrent do daemona
}, 500);

try {
  for await (const token of handle) {
    process.stdout.write(token);
  }
} catch (err) {
  if (err instanceof ChomatoCancelledError) {
    console.log("Generowanie zostało poprawnie anulowane.");
  }
}
```

---

### Complete End-to-End Example (Jak to wygląda w praktyce)

Oto pełny, realistyczny scenariusz użycia w edytorze kodu lub narzędziu CLI:

```typescript
import { ChomatoClient } from '@chomato/sdk';

async function main() {
  // 1. Połączenie z daemonem Chomato
  await using client = await ChomatoClient.connect({
    transport: 'unix',
    path: '/tmp/chomato.sock',
  });

  // 2. Przygotowanie/Indeksowanie wiedzy (Bloki w BlockStore)
  const sysPrompt = await client.blocks.index({
    id: 'system-prompt',
    content: 'Jesteś asystentem refaktoryzacji kodu.',
  });

  const fileContent = await client.blocks.index({
    id: 'file-main-ts',
    content: 'function add(a, b) { return a + b; }',
  });

  // 3. Budowanie głównego kontekstu
  const mainCtx = client.createContext()
    .attachBlock(sysPrompt)
    .attachBlock(fileContent);

  // 4. Tworzenie dwojga rozgałęzień (Eksperymenty refaktoryzacji)
  const branchTS = mainCtx.fork();
  branchTS.appendUser("Dodaj typy TypeScript do funkcji add.");

  const branchDoc = mainCtx.fork();
  branchDoc.appendUser("Dodaj JSDoc do funkcji add.");

  // 5. Generowanie dla gałęzi 1 (Strumieniowanie)
  console.log('--- Gałąź TypeScript ---');
  for await (const token of branchTS.generate()) {
    process.stdout.write(token);
  }

  // 6. Generowanie dla gałęzi 2
  console.log('\n--- Gałąź JSDoc ---');
  for await (const token of branchDoc.generate()) {
    process.stdout.write(token);
  }

  // 7. Czyszczenie bloku, którego już nie potrzebujemy
  await fileContent.unload();
}

main().catch(console.error);
```
