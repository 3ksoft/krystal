import { DenoFileSource } from "../packages/quant/src/gguf/source-deno.ts";
import { GgufReader } from "../packages/quant/src/gguf/reader.ts";
import { Lfm2Tokenizer } from "../packages/lfm2/src/tokenizer.ts";

const modelPath = Deno.args[0] ?? "./models/LFM2.5-1.2B-Instruct-F16.gguf";
console.log(`[deep-stats] Analizowanie słownika z: ${modelPath}...\n`);

const source = await DenoFileSource.open(modelPath);

try {
  const reader = await GgufReader.open(source);
  const tokenizer = new Lfm2Tokenizer(reader);

  const totalTokens = tokenizer.idToToken.length;

  const unmapGpt2Byte = (str: string) => {
    return str
      .replaceAll("Ġ", " ")
      .replaceAll("Ċ", "\n")
      .replaceAll("ĉ", "\t");
  };

  // KATEGORIE JĘZYKOWE & SKRYPTOWE
  let countSpecial = 0;
  let countByteFallback = 0;
  
  // Non-ASCII sub-categories
  let countLatinExtended = 0; // Europejskie znaki łacińskie (np. ą, ę, ü, é)
  let countCyrillic = 0;      // Cyrylica (rosyjski, ukraiński...)
  let countCJK = 0;           // Chiński, Japoński, Koreański
  let countArabicHebrew = 0;  // Arabski, Hebrajski
  let countEmojiOther = 0;    // Emoji i inne symbole Unicode

  // ASCII sub-categories
  let countSingleCharAscii = 0;
  let countWhitespaceRuns = 0;     // Spacje, wcięcia (np. "    ", "\n\n")
  let countPureDigits = 0;         // Sam cyfry (np. "123", "000")
  let countPureEnglishWords = 0;   // Słowa czysto alfabetyczne (np. "error", "server")
  let countCodeIdentifiers = 0;    // Snake_case, camelCase (np. "status_code", "userId")
  let countCodePunctuation = 0;    // Symbolika kodu/JSON (np. "==", "=>", "{}", "[]")
  let countMixedAscii = 0;         // Inne hybrydy ASCII

  // Histogram długości tokenów (w znakach)
  const lenHistogram: Record<string, number> = {
    "1 znak": 0,
    "2-4 znaki": 0,
    "5-8 znaków": 0,
    "9-12 znaków": 0,
    "13+ znaków": 0,
  };

  // Regexy do klasyfikacji
  const RE_BYTE_TOKEN = /^<0x[0-9A-Fa-f]{2}>$/;
  const RE_DIGITS = /^\d+$/;
  const RE_WHITESPACE = /^\s+$/;
  const RE_PURE_ENGLISH = /^[A-Za-z]+$/;
  const RE_CODE_IDENTIFIER = /^[A-Za-z0-9_\-$]+$/;
  const RE_PUNCTUATION = /^[\!\@\#\$\%\^\&\*\(\)\_\+\-\=\[\]\{\}\;\:\'\"\,\<\>\.\?\/\x5C\|\`\~]+$/;

  const RE_CYRILLIC = /[\u0400-\u04FF]/;
  const RE_CJK = /[\u4E00-\u9FFF\u3040-\u30FF\u3130-\u318F]/;
  const RE_ARABIC_HEBREW = /[\u0600-\u06FF\u0590-\u05FF]/;
  const RE_LATIN_EXTENDED = /[\u00C0-\u024F]/;

  for (let id = 0; id < totalTokens; id++) {
    const raw = tokenizer.idToToken[id] ?? "";
    const decoded = unmapGpt2Byte(raw);
    const trimmed = decoded.trim();

    // Długość tokena
    const len = decoded.length;
    if (len === 1) lenHistogram["1 znak"]++;
    else if (len <= 4) lenHistogram["2-4 znaki"]++;
    else if (len <= 8) lenHistogram["5-8 znaków"]++;
    else if (len <= 12) lenHistogram["9-12 znaków"]++;
    else lenHistogram["13+ znaków"]++;

    // 1. Special & Byte Fallbacks
    if (raw.startsWith("<|") || id === tokenizer.bos || id === tokenizer.eos) {
      countSpecial++;
      continue;
    }
    if (RE_BYTE_TOKEN.test(raw)) {
      countByteFallback++;
      continue;
    }

    // 2. Non-ASCII Script Classification
    if (RE_CYRILLIC.test(decoded)) {
      countCyrillic++;
      continue;
    }
    if (RE_CJK.test(decoded)) {
      countCJK++;
      continue;
    }
    if (RE_ARABIC_HEBREW.test(decoded)) {
      countArabicHebrew++;
      continue;
    }
    if (RE_LATIN_EXTENDED.test(decoded)) {
      countLatinExtended++;
      continue;
    }

    // 3. ASCII Breakdown
    if (/^[\x00-\x7F]+$/.test(decoded)) {
      if (decoded.length === 1) countSingleCharAscii++;

      if (RE_WHITESPACE.test(decoded)) {
        countWhitespaceRuns++;
      } else if (RE_DIGITS.test(trimmed)) {
        countPureDigits++;
      } else if (RE_PURE_ENGLISH.test(trimmed)) {
        countPureEnglishWords++;
      } else if (RE_CODE_IDENTIFIER.test(trimmed)) {
        countCodeIdentifiers++;
      } else if (RE_PUNCTUATION.test(trimmed)) {
        countCodePunctuation++;
      } else {
        countMixedAscii++;
      }
    } else {
      countEmojiOther++;
    }
  }

  const pct = (n: number) => ((n / totalTokens) * 100).toFixed(2) + "%";

  console.log("=================================================");
  console.log("     GŁĘBOKA ANALIZA STRUKTURY SŁOWNIKA (65.5K)  ");
  console.log("=================================================");
  
  console.log("\n1. STRUKTURA NIE-ASCII (WIELO JĘZYKI & SYMBOLE):");
  console.log(`   • CJK (Chiński/Japoński/Koreański): ${countCJK.toString().padStart(5)}  (${pct(countCJK)})`);
  console.log(`   • Cyrylica (Rosyjski/Ukraiński):   ${countCyrillic.toString().padStart(5)}  (${pct(countCyrillic)})`);
  console.log(`   • Arabski / Hebrajski:            ${countArabicHebrew.toString().padStart(5)}  (${pct(countArabicHebrew)})`);
  console.log(`   • Łaciński Rozszerzony (PL/DE/FR):${countLatinExtended.toString().padStart(5)}  (${pct(countLatinExtended)})`);
  console.log(`   • Emoji & Inne Symbole Unicode:   ${countEmojiOther.toString().padStart(5)}  (${pct(countEmojiOther)})`);
  console.log(`   • Byte Fallbacks (<0x..>):        ${countByteFallback.toString().padStart(5)}  (${pct(countByteFallback)})`);

  console.log("\n2. STRUKTURA ASCII (KOD / ANGIELSKI / JSON):");
  console.log(`   • Czyste Słowa Angielskie:        ${countPureEnglishWords.toString().padStart(5)}  (${pct(countPureEnglishWords)})`);
  console.log(`   • Identyfikatory (snake/camel):   ${countCodeIdentifiers.toString().padStart(5)}  (${pct(countCodeIdentifiers)})`);
  console.log(`   • Czyste Liczby / Cyfry:          ${countPureDigits.toString().padStart(5)}  (${pct(countPureDigits)})`);
  console.log(`   • Punctuation / Składnia Kodu:    ${countCodePunctuation.toString().padStart(5)}  (${pct(countCodePunctuation)})`);
  console.log(`   • Spacje / Wcięcia / Nowe Linie:  ${countWhitespaceRuns.toString().padStart(5)}  (${pct(countWhitespaceRuns)})`);
  console.log(`   • Inne hybrydy ASCII:             ${countMixedAscii.toString().padStart(5)}  (${pct(countMixedAscii)})`);
  console.log(`   • Kontrolne / Specjalne:          ${countSpecial.toString().padStart(5)}  (${pct(countSpecial)})`);

  console.log("\n3. HISTOGRAM DŁUGOŚCI TOKENÓW:");
  for (const [range, count] of Object.entries(lenHistogram)) {
    console.log(`   • ${range.padEnd(12)}: ${count.toString().padStart(5)}  (${pct(count)})`);
  }

  // PRZYKŁADOWY PROGNOZOWANY ROZMIAR PO PRZYCIĘCIU
  const strictCodeJsonVocab = 
    countSingleCharAscii + 
    countWhitespaceRuns + 
    countPureDigits + 
    countPureEnglishWords + 
    countCodeIdentifiers + 
    countCodePunctuation + 
    countSpecial + 
    countByteFallback;

  console.log("\n=================================================");
  console.log(`💡 PROGNOZA: Zbiór "Strict English + Code + JSON":`);
  console.log(`   Tylko ${strictCodeJsonVocab} tokenów (ok. ${pct(strictCodeJsonVocab)} całego słownika!)`);
  console.log(`   Możemy bezpiecznie odrzucić ~${totalTokens - strictCodeJsonVocab} tokenów!`);
  console.log("=================================================");



  // =================================================================
  // 4. GENERATOR MAPY RECYKLINGU TOKENÓW (1000 CUSTOM TOKENÓW)
  // =================================================================
  const RECYCLABLE_LIMIT = 1000;
  const victimTokens: Array<{ id: number; raw: string; decoded: string; category: string }> = [];

  for (let id = 0; id < totalTokens; id++) {
    if (victimTokens.length >= RECYCLABLE_LIMIT) break;

    const raw = tokenizer.idToToken[id] ?? "";
    const decoded = unmapGpt2Byte(raw);

    // Wybieramy tylko ewidentne nie-ASCII (np. CJK lub Cyrylicę), które na 100% nie wystąpią w JSON/Kodzie
    if (RE_CJK.test(decoded)) {
      victimTokens.push({ id, raw, decoded, category: "CJK" });
    } else if (RE_CYRILLIC.test(decoded)) {
      victimTokens.push({ id, raw, decoded, category: "Cyrillic" });
    }
  }

  console.log("\n=================================================");
  console.log(`🎯 ZNALEZIONO ${victimTokens.length} TOKENÓW-OFIAR DO RECYKLINGU:`);
  console.log("=================================================");
  console.log("Pierwsze 10 zrecyklingowanych slotów:");
  
  // Tworzymy mapę podmiany na customowe tokeny
  const customTokenMap: Record<string, number> = {};
  
  victimTokens.slice(0, 10).forEach((item, index) => {
    const customName = index === 0 
      ? "[--gpu-tool-call--]" 
      : index === 1 
      ? "[---end-gpu-tool-call--]" 
      : `[--identifier_${index - 1}--]`;
      
    customTokenMap[customName] = item.id;
    console.log(`   • ID: ${item.id.toString().padStart(5)} | Stary token: "${item.decoded}" (${item.category})  --> NOWY: ${customName}`);
  });

  // Zapisujemy mapę do pliku JSON dla Twojego silnika chomato / schema-pop
  const mapPath = "./custom-token-map.json";
  await Deno.writeTextFile(mapPath, JSON.stringify(customTokenMap, null, 2));
  console.log(`\nWygenerowano plik mapowania: ${mapPath}`);


} finally {
  source.close();
}


