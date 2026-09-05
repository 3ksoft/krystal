
# Dokumentacja Języka Krystal
*Interaktywny przewodnik po gramatyce i składni.*

UWAGA POLSKA PISOWNIAA JEST BŁĘDNA TRAKTOWAĆ TYLKO OBRAZOWO

---

## Składnia

Składnia Krystal została zaprojektowana z myślą o maksymalnej jednoznaczności i prostocie budowy zdań.

Fundamentalne zasady to:
* **Reguła Biernika:** Końcówka `-n` jednoznacznie identyfikuje dopełnienie.
* **Reguła Fokusu:** Pierwszy element w zdaniu jest tematem wypowiedzi.
* **Reguła Intencji:** Znak na końcu zdania definiuje jego cel.
* **Reguła Jawnego Podmiotu:** Każde zdanie musi posiadać jawnie określony podmiot. Konstrukcje bezosobowe (np. *„trzeba”*, *„należy”*) muszą być przeformułowane tak, aby podmiot był obecny.

---

## Intencja Zdania

Intencję zdania określa się za pomocą znaków interpunkcyjnych, które mogą być albo terminatorami (na końcu zdania), albo łącznikami (między zdaniami składowymi).

| Symbol | Typ | Znaczenie | Przykład |
| :--- | :--- | :--- | :--- |
| `.` | Terminator | Stwierdzenie faktu | `słońco świecis.` *(the sun shines.)* |
| `?` | Terminator | Pytanie | `ty byis śćęśliwa?` *(are you happy?)* |
| `!` | Terminator | Rozkaz | `hodi rapide!` *(come quickly!)* |
| `->` | Łącznik | Warunek rzeczywisty („jeśli... to...”) | `pados deśćo -> ja zostanios w domo.` |
| `~` | Łącznik | Warunek hipotetyczny („gdyby... to...”) | `ja mias kluczey ~ ja nie musias wracać.` |

---

## Rzeczownik

Rzeczownik nazywa osoby, miejsca, rzeczy i pojęcia. Odpowiada na pytania: *kto? co?*. W Krystal zawsze kończy się na **-o**.

#### Forma podstawowa rzeczownika
> `domo`, `vodo`, `ńebo`

### Odmiana
Rzeczownik odmienia się poprzez dodanie przyrostków do formy podstawowej. Obowiązuje ścisła kolejność morfemów: `RDZEŃ-FORMANT-PRZYPADEK-LICZBA`.

* **Plural:** `+y` – liczba mnoga
* **Accusative:** `+n` – biernik
* **Plural (Accusative):** `+n + y` – biernik + liczba mnoga

#### Pełna odmiana słowa „kobieta”
> * Mianownik l.poj: `kobieto`
> * Mianownik l.mn: `kobietoy`
> * Biernik l.poj: `kobieton`
> * Biernik l.mn: `kobietony`

---

## Czasownik

Czasownik nazywa czynności lub stany. Odpowiada na pytania: *co robi? co się z nim dzieje?*. W Krystal forma podstawowa jest jednocześnie formą w czasie teraźniejszym i kończy się na **-i**.

#### Forma podstawowa / czas teraźniejszy
> `miśo robi.`  
> `tato robi.`

### Aspekt (Dokonany / Niedokonany)
Krystal rozróżnia dwa aspekty czasownika: niedokonany (czynność trwająca lub powtarzająca się) i dokonany (czynność zakończona lub jednorazowa).

* **Aspekt niedokonany:** forma podstawowa czasownika kończąca się na `-i`.
* **Aspekt dokonany:** tworzy się go poprzez dodanie przedrostka **za-** do formy podstawowej.

#### Tworzenie aspektu dokonanego
> `robi` → `zarobi`  
> `kupi` → `zakupi`  
> `pisi` → `zapisi`  
> `ćyti` → `zaćyti`

### Odmiana (Koniugacja)
Odmiana czasowników jest w pełni regularna i nie zależy od osoby czy rodzaju. Polega na dodaniu odpowiedniej końcówki do rdzenia.

| Czas / Tryb | Końcówka | Przykład (rdzeń `rob-`) |
| :--- | :--- | :--- |
| Forma podstawowa / Czas teraźniejszy | `-i` | `robi` |
| Past Tense (Czas przeszły) | `-as` | `robias` |
| Future Tense (Czas przyszły) | `-os` | `robios` |

### Czasowniki Modalne
Czasowniki modalne, takie jak `muśi` (musieć), `povini` (powinien) i `mógi` (móc), są traktowane jak regularne czasowniki. Odmieniają się przez czasy i podlegają **Regule Sekwencji Czasowników**, co oznacza, że czasownik następujący po nich musi być w formie podstawowej (z końcówką `-i`).

#### Użycie czasowników modalnych
> *Present: I must go.*  
> `jao muśi iśi.`
>
> *Past: I should have done that.*  
> `jao povinias zrobii ton.`
>
> *Future: You will be able to ask.*  
> `tyo mogios zapyti.`

### Fokus i Strona Bierna
Dzięki jednoznacznemu biernikowi (końcówka **-n**), Krystal nie potrzebuje strony biernej. Tę samą funkcję realizuje się poprzez zmianę szyku wyrazów, co pozwala położyć fokus (nacisk) na dowolny element zdania, nie zmieniając jego logicznego znaczenia.

#### Zmiana fokusu w zdaniu
> `koto uovi myśon.`  
> *(Focus on the cat – it is the cat that performs the action)*

> `myśon uovi koto.`  
> *(Focus on the mouse – equivalent to the passive voice: "the mouse is caught by the cat")*

> `uovi koto myśon.`  
> *(Focus on the action of catching)*

### Sekwencje Czasowników i Bezokolicznik
Krystal nie ma oddzielnej formy bezokolicznika. Jego funkcję pełni forma podstawowa czasownika (z końcówką `-i`), gdy występuje on w sekwencji po innym, odmienionym już czasowniku.

**Reguła:** W sekwencji czasowników tylko pierwszy z nich może przyjmować końcówkę czasu (`-as`, `-os`). Wszystkie następne muszą być w formie podstawowej.

#### Użycie bezokolicznika
> `jao hći iśi.`  
> `jao mogias zaći pisi.`

### Imiesłowy i Opisywanie Stanów
Krystal nie posiada dedykowanych form gramatycznych dla imiesłowów (np. *„piszący”*, *„zrobiony”*). Zamiast tego stany i cechy wynikające z czynności wyraża się na dwa sposoby:

#### 1. Stan bez wyrażonego sprawcy (Pure State)
Aby opisać stan obiektu, gdy sprawca czynności jest nieznany lub nieistotny, traktujemy polski imiesłów bierny jako rdzeń dla zwykłego przymiotnika z końcówką **-a**.

> *EN: The door is closed.*  
> `dźvo bi zamknienta.`
>
> *EN: I see a running man.*  
> `jao vidźi biegnaca ćuviekon.`

#### 2. Stan z wyrażonym sprawcą (Focus Rule)
Gdy chcemy opisać stan obiektu i jednocześnie wskazać, kto wykonał czynność, unikamy strony biernej. Zamiast tego używamy standardowego zdania w stronie czynnej w połączeniu z **Regułą Fokusu**. Obiekt, na którym chcemy się skupić, umieszczamy na początku zdania.

> *EN: The system was started by the administrator. (Focus on the system)*  
> `systemon zauruhomias administratoro.`
>
> *Dla porównania – standardowe zdanie z fokusem na sprawcy:*  
> `administratoro zauruhomias systemon.`

---

## Przymiotnik

Przymiotnik opisuje cechy rzeczownika. Odpowiada na pytania: *jaki? jaka? jakie?*. W Krystal jest **nieodmienny** i zawsze kończy się na **-a**.

#### Forma podstawowa przymiotnika
> `dobra`, `duża`, `piekna`

### Brak odmiany
Przymiotnik w Krystal jest nieodmienny. Jego forma nigdy się nie zmienia, niezależnie od liczby czy przypadku rzeczownika, który opisuje.

> `duża domo.`  
> `duża domoy.`  
> `duża domon.`  
> `duża domony.`

### Stopniowanie Przymiotników
Stopniowanie odbywa się poprzez dodanie regularnych przedrostków:
* **Comparative Degree (Stopień wyższy):** `nad-`
* **Superlative Degree (Stopień najwyższy):** `naj-`
* **Equative Degree (Stopień równy):** `sam-`

#### Użycie stopniowania
> `samduża domo.`  
> `nadduża domo.`  
> `najduża domo.`

### Negacja Przymiotników
Przeciwieństwo przymiotnika tworzy się poprzez dodanie przedrostka **nie-**:
> `niedobra`

---

## Zaimek

Zaimek zastępuje w zdaniu rzeczownik, osobę lub inną część mowy. Odpowiada na pytania: *kto? co? jaki? który? czyj?*. W Krystal zaimki osobowe mają stałe rdzenie i odmieniają się tak samo jak rzeczowniki, przyjmując końcówkę **-o**.

### Zaimki Osobowe
Zaimki osobowe odmieniają się przez przypadki (mianownik i biernik), ale nie przez liczbę – każda forma liczbowa ma swój unikalny rdzeń.

#### Singular (Liczba pojedyncza)
| Znaczenie | Mianownik (kto? co?) | Biernik (kogo? co?) |
| :--- | :--- | :--- |
| *ja* | `jao` | `jaon` |
| *ty* | `tyo` | `tyon` |
| *on/ona/ono* | `ono` | `onon` |
| *to/tamto* | `to` | `ton` |

#### Plural (Liczba mnoga)
| Znaczenie | Mianownik (kto? co?) | Biernik (kogo? co?) |
| :--- | :--- | :--- |
| *my* | `myo` | `myon` |
| *wy* | `wyo` | `wyon` |
| *oni/one* | `onyo` | `onyon` |

### Zaimki Dzierżawcze
Zaimki dzierżawcze tworzy się w sposób regularny od formy biernika dowolnego zaimka lub rzeczownika, dodając końcówkę przymiotnika **-a**. Zgodnie z regułą dla przymiotników, są one nieodmienne.

#### Tworzenie zaimka dzierżawczego
> `jaon` → `jaona`  
> `miśon` → `miśona`

#### Użycie zaimka dzierżawczego
> `Widzis jaona domony.`  
> *(You see my houses. The possessive pronoun `jaona` remains unchanged.)*

### Zaimki Zwrotne
Krystal nie posiada odrębnego zaimka zwrotnego (*się*). Funkcję tę pełni forma biernika zaimka osobowego, gdy podmiot i dopełnienie w zdaniu są tą samą osobą lub rzeczą.

> `ono kohi onon.`  
> *(He loves himself.)*

### Zaimki Pytające i Względne
*(W budowie)*
* `kto` (who)
* `co` (what)
* `jaki` (what kind of)
* `który` (which)
* `gdzie` (where)
* `kiedy` (when)
* `jak` (how)
* `dlaczego` (why)

---

## Przysłówki i Partykuły

W Krystal wyróżniamy dwie główne grupy słów określających kontekst lub sposób czynności: przysłówki pochodne oraz stałą listę nieodmiennych przysłówków podstawowych i partykuł.

### 1. Przysłówki Pochodne (na -e)
Przysłówki pochodne opisują **sposób** wykonania czynności (*jak?*). Tworzy się je regularnie od przymiotników, zamieniając końcówkę **-a** na **-e**.

> `dobra` → `dobre`  
> `szybka` → `szybke`

### 2. Przysłówki Podstawowe i Partykuły (Nieodmienne)
Zamknięta lista nieodmiennych słów precyzujących kontekst logiczny, czasowy, przestrzenny lub częstotliwość. Nie przyjmują żadnych końcówek.

| Kategoria | Słowo | Znaczenie |
| :--- | :--- | :--- |
| **Logika i Fokus** | `także` | też, również |
| | `tylko` | tylko |
| | `navet` | nawet |
| | `dopiero` | dopiero (później niż oczekiwano) |
| | `już` | już (wcześniej niż oczekiwano) |
| | `jesće` | jeszcze (kontynuacja) |
| | `około` | około (aproksymacja) |
| **Czas** | `teraz` | teraz |
| | `vtedy` | wtedy |
| | `vćoraj` | wczoraj |
| | `jutro` | jutro |
| **Miejsce** | `tutaj` | tutaj |
| | `tam` | tam |
| **Częstotliwość** | `zavśe` | zawsze |
| | `nigdy` | nigdy |
| | `ćęsto` | często |

---

## Przyimki

Przyimki w Krystal służą do precyzyjnego wyrażania relacji przestrzennych, czasowych i logicznych.

### Reguła Ruchu vs. Lokalizacji
1. Gdy przyimek opisuje **lokalizację (gdzie?)**, następujący po nim rzeczownik jest w **formie podstawowej**.
2. Gdy przyimek opisuje **ruch (dokąd?)**, następujący po nim rzeczownik przyjmuje **końcówkę biernika (-n)**.

#### Przykład: `v` (w / do)
> **Lokalizacja:** `jao byi v domo.` *(I am in the house.)*  
> **Ruch:** `jao idzi v domon.` *(I am going into the house.)*

#### Przykład: `na` (na / na)
> **Lokalizacja:** `knigo leżi na stoło.` *(The book is on the table.)*  
> **Ruch:** `jao kładzi knigon na stołon.` *(I am putting the book onto the table.)*

### Zastępowanie Polskich Przypadków
* **Dopełniacz (kogo? czego?)**
  * Przynależność: `domo tatona` (dom taty) – użycie przymiotnika dzierżawczego.
  * Część z całości: `kavauo od torto` (kawałek tortu).
  * Wymaganie czasownika: `jao potśebui vodon` (potrzebuję wody) – czasownik rządzi biernikiem.
* **Celownik (komu? czemu?)**
  * Użycie przyimków `dla` lub `do`.
  * `jao kupias prezento dla ty.` (Kupiłem prezent dla ciebie.)
  * `jao dai knigon do ty.` (Daję książkę tobie.)
* **Narzędnik (kim? czym?)**
  * Użycie przyimka `z`.
  * `jao pisi z ołóvkon.` (Piszę ołówkiem.)
* **Miejscownik (o kim? o czym?)**
  * Obsługiwany przez przyimki lokalizacji (`o`, `v`, `na`, `przy`).
  * `myo myśli o ty.` (Myślimy o tobie.)

Podstawowe przyimki to: `v`, `na`, `z`, `do`, `od`, `dla`, `o`, `przy`, `nad`, `pod`, `pśed`, `podćas`.

---

## Spójniki

Podstawowe spójniki to: `i`, `albo`, `ale`, `że`, `jeśli`.

#### Example C-1: Użycie spójników
> `koto i pso.` *(cat and dog.)*  
> `ja wiem, że ty tam byis.` *(I know that you are there.)*

### Filozofia i Upraszczanie
Krystal unika złożonych spójników podrzędnych (*chociaż, mimo że, zanim, dopóki*). Stosuje się dwie strategie:
1. **Użycie prostych spójników:** `ale` (dla kontrastu), `i` (dla sekwencji zdarzeń).
2. **Przeformułowanie zdania:** Użycie operatorów `->` (*jeśli...to*) lub `~` (*gdyby...to*).

---

## Negacja

### Reguła Negacji
Słowo **nie** zawsze występuje bezpośrednio przed czasownikiem, którego dotyczy. Nie ma podwójnych zaprzeczeń.

#### Użycie „nie”
> Zdanie twierdzące: `jao vidźi koton.`  
> Zdanie przeczące: `jao nie vidźi koton.`

---

## Zaimki Względne i Korelatywy

Zdania względne są wprowadzane przez korelatywy oparte na rdzeniu **k-**.

### Tabela Korelatywów
| Znaczenie | Forma Bazowa | Przykład Użycia |
| :--- | :--- | :--- |
| person | `kto` | `ćuovieko, kto...` |
| thing | `ko` | `kśażko, ko...` |
| place | `kieo` | `domo, kieo...` |
| time | `kiempo` | `dźeńo, kiempo...` |
| quality/kind | `kia` | `sposobo, kia...` |
| reason | `kio` | `pśyćyno, kio...` |

### Odmiana
Korelatywy pełniące funkcję dopełnienia (`kto`, `ko`, `kieo`, `kia`) przyjmują w bierniku końcówkę `-n`. **Nie odmieniają się przez liczbę** – liczba wynika z rzeczownika odniesienia.

> `domo, kieon jao vidźi` *(the house, which [as a place] I see)*  
> `metodo, kian jao preferui` *(the method, which I prefer)*

### Zaimki Dzierżawcze Względne („którego/której”)
1. Korelatyw podstawowy: `kto` / `ko`
2. Forma biernika: `kton` / `kon`
3. Końcówka przymiotnika `-a`: `ktona` / `kona`

> `ćłowieko, ktona domon jao vidźas.` *(the person, whose house I saw)*  
> `książko, kona okładko byi zńiśćona.` *(the book, whose cover is damaged)*

### Składnia Zdania Względnego
```text
[MAIN CLAUSE], [CORRELATIVE + SUBORDINATE CLAUSE].