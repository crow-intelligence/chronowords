# Hansard: a century of House of Commons debates, decade by decade

Does chronowords do what it is for — detecting semantic change — at the scale
it now promises, on a real diachronic corpus, on a laptop? This experiment
trains one model per decade on every House of Commons debate from 1919 to
October 2026 and aligns consecutive decades.

## Data

- **Source:** the House of Commons debates XML published by TheyWorkForYou /
  ParlParse (mySociety), mirrored untouched with `scripts/fetch_hansard.sh`
  (rsync; 20,159 debate files, 8.5 GB; manifest in
  `data/raw/hansard/debates.manifest.json`).
- **Licence:** *Contains Parliamentary information licensed under the Open
  Parliament Licence v3.0.*
- **Extraction** (`scripts/prepare_hansard.py`): one version per sitting day
  (the file marked `latest="yes"`), and the text of every `<p>` inside a
  `<speech>`, one paragraph per line. Headings, tables and division lists are
  left out. The text stays raw (case and punctuation kept); tokenisation
  happens at training time. 15,543 sitting days; **0 files failed to parse**.
  Per-decade records are in `results/corpora/`.
- The source has fewer sitting days for the 2000s (969) than for neighbouring
  decades (~1,500), and the 1910s cover only 1919. Both are reflected as-is.

## Method

`scripts/run_hansard.py train <decade>` trains `SVDAlgebra` with
`tokenizer="unicode"` (ICU4X) and `lowercase=True`; everything else is a
library default: 300 dimensions, `window_size=5`, `min_word_length=3`,
`vocab_size=50_000`, `min_count=100`, PPMI as published by Levy, Goldberg &
Dagan (2015) with `alpha=0.75` and `ppmi_min_count=1`, and `eig=0.5`. The PPMI
defaults were chosen with `experiments/evaluation`. Each decade runs in its own
process, so peak memory is per decade. `analyse` aligns each pair of
consecutive decades with `ProcrustesAligner`, scores 19 target words by
*shift = 1 − aligned cosine similarity*, and lists each word's nearest
neighbours per decade. A word below `min_count` in a decade has no vector
there. Words without a vector (no PPMI entries) are skipped; with these
defaults there were none among the targets.

Machine: 4-core laptop, 25 GB RAM, Python 3.13, counting on 4 threads.

## Performance

| decade | sitting days | text | tokens | vocabulary | PPMI entries | training time | peak memory |
|---|---|---|---|---|---|---|---|
| 1910s | 159 | 45 MB | 6.1 M | 4,085 | 972,322 | 7 s | 0.29 GB |
| 1920s | 1,093 | 321 MB | 43.6 M | 11,422 | 3,985,886 | 25 s | 0.73 GB |
| 1930s | 1,520 | 441 MB | 59.7 M | 13,296 | 4,999,416 | 33 s | 0.82 GB |
| 1940s | 1,331 | 394 MB | 53.2 M | 12,879 | 4,834,605 | 32 s | 0.71 GB |
| 1950s | 1,516 | 511 MB | 69.1 M | 14,666 | 5,769,709 | 44 s | 0.85 GB |
| 1960s | 1,632 | 604 MB | 81.2 M | 16,116 | 6,568,326 | 42 s | 1.01 GB |
| 1970s | 1,607 | 609 MB | 81.7 M | 16,557 | 6,809,701 | 43 s | 1.10 GB |
| 1980s | 1,681 | 651 MB | 87.0 M | 17,801 | 7,636,795 | 47 s | 1.02 GB |
| 1990s | 1,572 | 604 MB | 80.8 M | 17,550 | 7,540,897 | 50 s | 1.01 GB |
| 2000s | 969 | 361 MB | 48.2 M | 13,611 | 5,265,771 | 36 s | 0.88 GB |
| 2010s | 1,455 | 563 MB | 75.6 M | 16,742 | 6,919,248 | 43 s | 1.13 GB |
| 2020s | 1,008 | 388 MB | 52.1 M | 14,592 | 5,806,055 | 33 s | 0.90 GB |
| **all** | **15,543** | **5.49 GB** | **738 M** | | | **7.2 min** | max 1.13 GB |

All 12 models used the sparse SVD path. Results: `results/train.jsonl`, with
the full `training_record` of every model. Compared with the first run of this
experiment, which used the old PPMI thresholds (`ppmi_min_count=5`), training
takes 7.2 instead of 4.5 minutes and up to 1.13 instead of 1.03 GB. The PPMI
matrices keep far more pairs.

## What moved

Mean and maximum shift between consecutive decades. *house* and *water*
are controls whose meaning did not change:

| word | decade pairs | mean shift | max shift |
|---|---|---|---|
| gay | 4 | 0.49 | 0.54 |
| wireless | 8 | 0.46 | 0.78 |
| broadcast | 10 | 0.29 | 0.49 |
| computer | 6 | 0.32 | 0.40 |
| web | 5 | 0.62 | 0.78 |
| mobile | 10 | 0.50 | 0.67 |
| cell | 9 | 0.51 | 0.62 |
| network | 9 | 0.29 | 0.57 |
| virus | 5 | 0.57 | 0.72 |
| green | 11 | 0.39 | 0.97 |
| environment | 10 | 0.30 | 0.59 |
| climate | 10 | 0.31 | 0.51 |
| terror | 10 | 0.49 | 0.73 |
| immigration | 10 | 0.33 | 0.53 |
| europe | 11 | 0.11 | 0.32 |
| empire | 11 | 0.36 | 0.52 |
| commonwealth | 10 | 0.23 | 0.36 |
| house | 11 | 0.08 | 0.17 |
| water | 11 | 0.08 | 0.17 |

The controls stay low (0.08 on average), lower than with the old thresholds
(0.10 and 0.17). Words with documented changes in meaning or reference move
three to eight times as much. The neighbours show what moved:

| decade | nearest neighbours of **gay** |
|---|---|
| 1980s | lesbian, homosexuals, homosexual, lesbians, homosexuality, heterosexual |
| 1990s | homosexual, heterosexual, consenting, homosexuals, sexuality, intercourse |
| 2000s | lesbian, homosexual, discriminate, sex, married, homosexuality |
| 2010s | lesbian, bisexual, lgbt, heterosexual, marriage, sex |
| 2020s | lesbian, lesbians, bisexual, lgbt, trans, transgender |

| decade | nearest neighbours of **cell** |
|---|---|
| 1930s | cells, warders, prisoner, locked, prison |
| 1940s | cells, confinement, prison, bedroom, holloway |
| 1950s | cells, prisons, prison, sleeping, prisoners |
| 1960s | cells, solitary, confinement, wormwood, holloway |
| 1970s | cells, prisoners, maze, prisons, prison |
| 1980s | cells, prisoners, inmates, scrubs, prison |
| 1990s | cells, strangeways, gene, prisons, inmates |
| 2000s | stem, embryonic, cells, embryo, embryos |
| 2010s | transplant, transplants, cord, organ, transplantation |
| 2020s | battery, transplant, leukaemia, chemotherapy, kidney |

| decade | nearest neighbours of **virus** |
|---|---|
| 1950s | infected, infection, mouth, fever, carcases |
| 1960s | outbreak, mouth, infection, epidemic, outbreaks |
| 1980s | hiv, aids, infection, infected, needles |
| 1990s | haemophiliacs, hepatitis, hiv, haemophilia, infection |
| 2000s | avian, flu, swine, influenza, bluetongue |
| 2010s | hpv, vaccine, vaccinated, ebola, vaccination |
| 2020s | omicron, infection, covid, variant, variants |

| decade | nearest neighbours of **climate** |
|---|---|
| 1920s | soil, grow, fertile, ours, climatic |
| 1930s | soil, climatic, weather, crops, grass |
| 1940s | weather, climatic, frost, storms, tropical |
| 1950s | climatic, prevailing, weather, conditions, wet |
| 1960s | atmosphere, conducive, flourish, attitudes, mood |
| 1970s | atmosphere, flourish, economy, create, thrive |
| 1980s | atmosphere, flourish, prosper, economy, stimulate |
| 1990s | atmosphere, warming, rio, global, kyoto |
| 2000s | change, warming, global, kyoto, deforestation |
| 2010s | warming, 2050, emissions, global, carbon |
| 2020s | warming, planet, cop26, adaptation, global |

| decade | nearest neighbours of **web** |
|---|---|
| 1970s | perils, cordial, teething, harmonious, roost |
| 1980s | bureaucratic, bureaucracy, hedged, freed, relaxing |
| 1990s | internet, website, networks, tape, interactive |
| 2000s | internet, websites, online, images, website |
| 2010s | internet, facebook, websites, online, twitter |
| 2020s | algorithms, regulating, websites, facebook, internet |

| decade | nearest neighbours of **terror** |
|---|---|
| 1920s | terrorism, outrage, awful, horror, reprisals |
| 1930s | horror, fear, dread, despair, hearts |
| 1940s | horror, agony, tyranny, sufferings, sorrow |
| 1950s | terrorism, mob, anger, despair, bitterness |
| 1960s | horror, stalemate, vietcong, vietnam, victory |
| 1970s | killings, terrorism, violence, i.r.a, terrorists |
| 1980s | terrorism, terrorist, killings, terrorists, atrocities |
| 1990s | intimidation, brutal, killings, beatings, torture |
| 2000s | terrorist, qaeda, terrorists, terrorism, taliban |
| 2010s | terrorist, terrorism, terrorists, qaeda, islamist |
| 2020s | terrorist, terrorism, hezbollah, hamas, islamist |

Every target word's neighbours in every decade are in `results/shifts.json`.
Two observations that were not targeted:

- *climate* shows three senses in turn: weather (1940s–50s), the metaphorical
  "climate" of *atmosphere, mood* (1960s–80s), then *Kyoto, warming* (from the
  1990s).
- *empire* moves from *dominions, Imperial Preference* to colonial history,
  to the Soviet bloc in the 1990s, to *Windrush* in the 2020s.

## Caveats (read before citing)

- **This is parliamentary language.** Neighbours reflect how the Commons
  talked, not English at large.
- **Proper names and set phrases dominate some words.** *green* is mostly
  *Bethnal Green* and, from the 1960s, *green belt*, so its high shift is not
  about colour or politics.
- **One model per decade, no confidence intervals.** Shift scores between
  consecutive decades carry SVD and sampling noise of a size this experiment
  does not measure; compare the controls before reading much into small
  differences.
- **`min_count=100` per decade** hides early, rare uses: *gay* only has a
  vector from the 1980s.
- **The source has gaps:** 969 sitting days for the 2000s, against about 1,500
  in neighbouring decades.

## Reproduce

```bash
experiments/hansard/scripts/fetch_hansard.sh                       # ~15 GB incl. svn metadata
uv run python experiments/hansard/scripts/prepare_hansard.py       # ~5.2 GB of text
for d in 1910s 1920s 1930s 1940s 1950s 1960s 1970s 1980s 1990s 2000s 2010s 2020s; do
    uv run python experiments/hansard/scripts/run_hansard.py train $d
done
uv run python experiments/hansard/scripts/run_hansard.py analyse
```
