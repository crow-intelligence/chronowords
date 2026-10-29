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
`tokenizer="unicode"` (ICU4X) and `lowercase=True`. Everything else is a library
default: 300 dimensions, `window_size=5`, `min_word_length=3`,
`vocab_size=50_000`, `min_count=100`, `ppmi_min_count=5`. Each decade runs in
its own process, so peak memory is per decade. `analyse` aligns each pair of
consecutive decades with `ProcrustesAligner`, scores 19 target words by
*shift = 1 − aligned cosine similarity*, and lists each word's nearest
neighbours per decade. A word below `min_count` in a decade has no vector
there. Words whose vector is all zero (no PPMI entries) are skipped, because
their shift is undefined.

Machine: 4-core laptop, 25 GB RAM, Python 3.13, counting on 4 threads.

## Performance

| decade | sitting days | text | tokens | vocabulary | PPMI entries | training time | peak memory |
|---|---|---|---|---|---|---|---|
| 1910s | 159 | 45 MB | 6.1 M | 4,085 | 47,237 | 4 s | 0.24 GB |
| 1920s | 1,093 | 321 MB | 43.6 M | 11,422 | 203,752 | 17 s | 0.53 GB |
| 1930s | 1,520 | 441 MB | 59.7 M | 13,296 | 256,926 | 20 s | 0.67 GB |
| 1940s | 1,331 | 394 MB | 53.2 M | 12,879 | 227,846 | 18 s | 0.62 GB |
| 1950s | 1,516 | 511 MB | 69.1 M | 14,666 | 276,635 | 25 s | 0.72 GB |
| 1960s | 1,632 | 604 MB | 81.2 M | 16,116 | 307,286 | 29 s | 0.63 GB |
| 1970s | 1,607 | 609 MB | 81.7 M | 16,557 | 307,575 | 29 s | 0.69 GB |
| 1980s | 1,681 | 651 MB | 87.0 M | 17,801 | 346,231 | 31 s | 0.99 GB |
| 1990s | 1,572 | 604 MB | 80.8 M | 17,550 | 340,955 | 29 s | 1.03 GB |
| 2000s | 969 | 361 MB | 48.2 M | 13,611 | 242,211 | 22 s | 0.60 GB |
| 2010s | 1,455 | 563 MB | 75.6 M | 16,742 | 345,906 | 27 s | 0.74 GB |
| 2020s | 1,008 | 388 MB | 52.1 M | 14,592 | 306,041 | 21 s | 0.67 GB |
| **all** | **15,543** | **5.49 GB** | **738 M** | | | **4.5 min** | max 1.03 GB |

All 12 models used the sparse SVD path (no noisy fallback). Results:
`results/train.jsonl`, with the full `training_record` of every model.

## What moved

Mean and maximum shift between consecutive decades. *house* and *water*
are controls whose meaning did not change:

| word | decade pairs | mean shift | max shift |
|---|---|---|---|
| gay | 4 | 0.64 | 0.81 |
| wireless | 8 | 0.62 | 0.92 |
| broadcast | 10 | 0.42 | 0.72 |
| computer | 6 | 0.58 | 0.69 |
| web | 5 | 0.82 | 0.97 |
| mobile | 10 | 0.76 | 1.05 |
| cell | 9 | 0.59 | 0.83 |
| network | 9 | 0.48 | 0.94 |
| virus | 5 | 0.74 | 1.03 |
| green | 11 | 0.53 | 1.10 |
| environment | 9 | 0.38 | 0.79 |
| climate | 10 | 0.55 | 0.82 |
| terror | 8 | 0.58 | 0.79 |
| immigration | 10 | 0.53 | 0.90 |
| europe | 11 | 0.21 | 0.43 |
| empire | 11 | 0.64 | 0.94 |
| commonwealth | 10 | 0.45 | 0.65 |
| house | 11 | 0.10 | 0.15 |
| water | 11 | 0.17 | 0.29 |

The controls stay low (0.10 and 0.17 on average). Words with documented
changes in meaning or reference move four to eight times as much. The
neighbours show what moved:

| decade | nearest neighbours of **gay** |
|---|---|
| 1980s | lesbians, luncheon, lesbian, bowling, men's, cosy |
| 1990s | heterosexual, homosexual, homosexuals, consenting, lesbian, unprotected |
| 2000s | lesbian, separating, homosexual, married, couples, marriage |
| 2010s | lesbian, stonewall, openly, straight, heterosexual, bisexual |
| 2020s | lesbian, lesbians, openly, bisexual, uganda, lgbtq |

| decade | nearest neighbours of **cell** |
|---|---|
| 1930s | locked, warders, prisoner, cells, prison |
| 1940s | holloway, prison, convict, escaped, prisons |
| 1950s | prisoners, cells, gaol, prisons, prisoner |
| 1960s | wormwood, prisoners, scrubs, cells, escaped |
| 1970s | prisoners, cells, maze, chopping, defaulters |
| 1980s | cells, stripped, inmates, prisoners, 3,500 |
| 1990s | cells, prisoners, wolds, inmates, strangeways |
| 2000s | stem, embryos, differentiate, cloning, therapeutic |
| 2010s | organ, organs, donation, transplant, stem |
| 2020s | gene, therapies, talking, radiation, behavioural |

| decade | nearest neighbours of **virus** |
|---|---|
| 1950s | ravages, eradicate, foul, endemic, killer |
| 1960s | symptom, leprosy, cured, sufferers, identifying |
| 1980s | hiv, aids, spread, navigational, contracted |
| 1990s | haemophiliacs, hepatitis, infection, haemophilia, symptomatic |
| 2000s | pandemic, flu, strain, preparedness, complications |
| 2010s | vaccine, vaccines, cow, vaccination, badgers |
| 2020s | viral, variant, transmission, covid, omicron |

| decade | nearest neighbours of **climate** |
|---|---|
| 1920s | soil, cultivators, productivity, tropical, jamaica |
| 1930s | cultivators, tilling, climatic, fertility, soil |
| 1940s | soil, irrigation, fertility, erosion, stored |
| 1950s | roots, fertility, samples, soil, temperate |
| 1960s | difference, differences, divergence, consensus, shade |
| 1970s | survive, thrive, soil, discouraging, economy |
| 1980s | prosper, rhetorical, thrive, enabled, flourish |
| 1990s | rio, kyoto, global, binding, buenos |
| 2000s | warming, mitigation, change, global, emissions |
| 2010s | warming, targets, ambitious, 2050, emissions |
| 2020s | gear, adaptation, change, aligned, mitigation |

| decade | nearest neighbours of **mobile** |
|---|---|
| 1920s | irresistible, lucidity, disciplined, force, unified |
| 1930s | batteries, battery, anti, antiaircraft, tanks |
| 1940s | n.a.a.f.i, rescue, demolition, canteens, canteen |
| 1950s | formations, formed, amalgamate, radiography, units |
| 1960s | vans, static, sheds, counters, grocers |
| 1970s | sites, gipsy, pitch, static, designs |
| 1980s | caravan, tent, caravans, gipsy, camping |
| 1990s | phones, hotline, telephone, tapping, interception |
| 2000s | phone, colours, telecommunications, siting, masts |
| 2010s | masts, computers, nowadays, picks, colours |
| 2020s | somebody’s, phone, switched, clearest, phones |

| decade | nearest neighbours of **web** |
|---|---|
| 1970s | helmet, contravene, conforming, apprised, regulations |
| 1980s | panoply, bribery, relaxing, corruption, stricter |
| 1990s | malpractice, exposing, endemic, corruption, dishonesty |
| 2000s | prioritise, suitability, designating, sites, traveller |
| 2010s | online, accessed, internet, pills, publishers |
| 2020s | computing, artificial, quantum, harness, embracing |

Every target word's neighbours in every decade are in `results/shifts.json`.

## Caveats (read before citing)

- **This is parliamentary language.** Neighbours reflect how the Commons
  talked: *empire* sits with tariffs and wines in the 1930s (Imperial
  Preference), with medals later (the British Empire Medal) and with
  *Windrush* in the 2020s.
- **Proper names dominate some words.** *green* is mostly *Bethnal Green*,
  plus the idiom "tightening the belt", so its high shift is not about
  colour or politics.
- **Some neighbourhoods are unexplained.** *computer* in the 2020s sits with
  *substance misuse* (perhaps the Computer Misuse Act). That was not checked
  against the text.
- **One model per decade, no confidence intervals.** Shift scores between
  consecutive decades carry SVD and sampling noise of a size this experiment
  does not measure; compare the controls before reading much into small
  differences.
- **`min_count=100` per decade** hides early, rare uses: *gay* only has a
  vector from the 1980s.

## Reproduce

```bash
experiments/hansard/scripts/fetch_hansard.sh                       # ~15 GB incl. svn metadata
uv run python experiments/hansard/scripts/prepare_hansard.py       # ~5.2 GB of text
for d in 1910s 1920s 1930s 1940s 1950s 1960s 1970s 1980s 1990s 2000s 2010s 2020s; do
    uv run python experiments/hansard/scripts/run_hansard.py train $d
done
uv run python experiments/hansard/scripts/run_hansard.py analyse
```
