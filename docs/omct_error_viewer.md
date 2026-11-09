# OMCT dictionary error viewer — V1 spec (CS-58)

Status: **implemented**. V1 reads the dictionary round's own artifacts
(CS-48 … CS-57); it runs **no** experiment, writes nothing under `results/`,
changes no log or auto table, and touches none of the UPR viewer's files. OMCT
is **auxiliary** data: every number here annotates method choice and is **never
a UPR metric**. No `upr-` run is listed.

## Scope and staging

| stage | what | status |
|---|---|---|
| **V1** | the dictionary's out-of-fold pair scores, rebuilt from the run artifact; four views; the guard; the control strip | this spec |
| V2 | pack / LLM views | not planned here |

V1 is **discovery instrumentation**, not a demo. It answers: which reviewed
pair did the dictionary rank badly, with which words, on which question, under
which arm, and how far did the floor move it?

Two facts fix the design.

1. **The judged score is out-of-fold.** `dictionary.tables` is the full map
   (all 1,520 pairs); a judged pair's stored score comes from the map of the
   fold that did **not** contain it. The viewer rebuilds the five fold maps
   with the round's own pure functions and reproduces every judged score — and
   every rank percentile — at **1e-12**. A mismatch is a guard failure, never
   a silent re-scoring.
2. **The rank is measured on the assembled matrix.** The pair's own cell is
   the fold map's; the paragraph's *other* candidates keep the **full** map's
   score. That is the matrix the artifact's `rank_percentile` was measured on,
   and the viewer rebuilds it the same way.

## The unit (the viewer's labels)

- **1,520 reviewer-judged pairs**: 524 positive, 996 reviewer-verified hard
  negatives (subset A). 225 paragraphs in 3 documents — NU 75 paragraphs /
  243 questions, CIDH 112 / 217, AI 38 / 51.
- 294 distinct question texts; 44,467 same-document cells.
- **4,000 unjudged control cells** (subset B). They are never ground truth,
  never in an error list, never pooled with the judged set. They appear **only**
  in the Errors view's separate control strip, with no label column.

Source: `data/external/omct_pairs.json` (the path in `configs/base.yaml`'s
`paths.data_external`). Each record has `pair_id`, `document`,
`paragraph_number`, `paragraph_text`, `question_number`, `question_text`,
`label`. Every artifact row joins on `pair_id`.

## Sources

| source | what V1 reads | provenance it checks |
|---|---|---|
| `data/external/omct_pairs.json` | the reviewed pairs (read-only) | sha256; `n_pairs` / `n_positive` / `n_negative` = 1520 / 524 / 996; 3 documents |
| `results/<run>/analysis.json` | `dictionary.map`, `dictionary.corpus`, `dictionary.tables`, `dictionary.words`, `dictionary.classes` (when present), `dictionary.availability`, `dictionary.readout`, `models[kind == "dictionary"].arms.dictionary.judged_pairs` / `control_cells`, the model's `readout.scorings` | `run_id` == the folder; seed 42; folds 5; 1,520 judged rows; 4,000 control cells; the map's counts |
| `results/omct-analyze-voyage-dict-abstain/analysis.json` | `readout.abstention_arms`: `floor_pairs`, `floor_negative_share`, `non_floor` | 518 / 0.7780 / 1002, checked **when the abstain run is selected** |
| `results/omct-analyze-voyage-dict-stability/analysis.json` | `dictionary.stability.links` (27,979 rows, the file is 60 MB) | read only when the stability run is selected; its links are indexed only when the One question view opens |

The offered runs are `omct-analyze-voyage-dict-z` (default, CS-50),
`…-dict` (CS-49), `omct-analyze-dictmap` (CS-48), `…-dict-phrases`,
`…-dict-stability`, `…-dict-lexicon`, `…-dict-abstain`, `…-dict-qpairs`. The
CS-48 run is **map-only**: it carries no dictionary scorer, so the viewer names
it and refuses the views with a clear message (it does not fake a scorer).

## The trap: the fold rebuild

`commands/omct.py`'s `_dictionary_fold_ids` is
`np.random.default_rng(seed).permutation(n_pairs)`, split into five contiguous
chunks (seed 42). The layer rebuilds, for each fold, a map from the **other**
folds with `features.question_map.build_question_map`, scores the whole
225 × 294 grid with `lexical.question_map_score.score_grid`, takes the pair's
own fold cell, and assembles the matrix: the judged cells from their folds, the
unjudged cells from the full map. `score_grid`'s token sum is over a **sorted**
order and is kept exactly.

The rebuild is **attribution and guard only**. It is labelled "recomputed
attribution" wherever it appears; it never becomes a new artifact number or a
new metric. The guard's check is the agreement with the artifact at 1e-12; on
the default run the maxima are both **0.0** (1,520 / 1,520 pairs).

The **contributing words** are the fold map's own rows for the pair's
paragraph: `word`, `n_with`, `df`, `log_odds`, `z_score`, and the additive
contribution under the selected arm (`sum_positive` keeps positive log-odds,
`sum_zscore_positive` positive z-scores, `support_count` one per present
table unit). They sum to the pair's score under the floor fill; a non-floor
fill replaces a **floored** pair's score and has no words behind it.

## The four views

### 1. One pair (the core)

The paragraph, the question, the human verdict, the artifact score, the
rebuilt score, the pair's rank among its paragraph's same-document candidates
(the run's scope, stable ties), the rank percentile, and whether the pair is
**FLOORED** — the question has no table in that fold's map — with the reason:
how many positives and how many reviewed pairs the question had in that fold.

A **floor / neutral / max** toggle (CS-56) shows the pair's rank move; the
floor fill is the artifact's own, the other two are recomputed. An **arm**
selector offers the run's own `readout.scorings` arms; only the primary arm's
judged pairs are in the artifact, so a non-primary arm is labelled
"recomputed attribution" and its score is not compared with the artifact's.

The word table is **signed**: every paragraph word in the question's fold
table, with `n_with`, `df`, the signed log-odds, the signed `z`, the arm's
contribution, a **positive / negative** direction and whether the arm actually
adds it. The direction follows the arm's own measure — log-odds for
`sum_positive`, `z` for the z arms; `support_count` has no sign of its own, so
its direction is the `z` sign and it adds every present word. **Negative words
are kept and shown**; a positive-only arm does not subtract them (they
contribute 0), which is why the same word can be positive for `sum_positive`
and negative for `sum_zscore_positive` — CS-50's sign reversal. A filter shows
all / positive / negative words.

### 2. Errors

The pre-registered **evidence cut**: a pair with `score > 0` is a positive
prediction (the map gave it points). The classes are:

| `error` | rule |
|---|---|
| `floored` | the question has no table in the pair's fold map — **its own class**, never folded into the false negatives |
| `true_positive` | label 1, not floored, score > 0 |
| `false_negative` | label 1, not floored, score ≤ 0 |
| `false_positive` | label 0, not floored, score > 0 |
| `true_negative` | label 0, not floored, score ≤ 0 |

The view filters by document, what to show (the errors / floored / all
classes), human verdict, **floored / non-floored** and sort (score or rank), and
opens a selected pair in One pair. The **floored positives** are counted separately. The **control
strip** at the bottom lists the 4,000 unjudged cells (no label column) and says
they are never an error, never ground truth, never pooled.

### 3. One question

The **full map's** table (top words by z, with `n_with`, `df`, lift, log-odds,
`z`, standard error), the question's reviewed-pair / positive / negative counts,
per-fold availability (pairs, positives, the fold map's table size, the
artifact's `with_table` / `with_evidence`, floored pairs), and the word
classes.

A word's class is **derived from `dictionary.words`' own `tables` count** with
`lexical.word_class.classify_word`: distinctive ≤ 2, shared 3–52, generic ≥ 53
of the 209 labelled tables (a quarter, so the boundary follows the corpus's
size). **No POS is invented**: the artifacts store no grammatical class per
word, and the view shows none.

When the stability run is selected, each link also carries `present_share`,
`sign_stability`, `head_stability` and its survival at τ = 0.75 / 0.90
(`sign_stability ≥ τ`). The 60 MB artifact is read when that run is selected
(the run's own scores need it); its links are indexed only when this view
opens, and the default run never reads it.

### 4. Compare arms

Two `(run, arm)` series over the same 1,520 pairs, joined on `pair_id`: the
score, the rank, the rank percentile and the **rank movement** (`Δ` is right
minus left; a positive `Δ rank` means the right series ranks the pair lower).
The artifact's own `within_auc` / `pooled_auc` / `floor_pairs` sit beside each
side. Scores are not compared across units (word vs phrase); the ranks are.

## The guard

`python tools/omct_error_view.py --check` prints a report naming every source
and exits **0** only if every check passes; on a mismatch it prints the
problem and exits **1**. The app runs the same function at load; on failure it
shows a red banner and blocks every view (`st.stop`).

1. **Run shape.** `run_id` == its folder; seed 42; `cross_fit.folds` 5; 1,520
   judged rows; 4,000 control cells; 3 documents; no control cell carries a
   label.
2. **Map size.** 294 questions, 209 with a table, 85 empty; every table's rows
   sum to `n_stored_pairs`; `n_words` equals `dictionary.words`' length. For
   the **word unit** the pins are **27,979** stored pairs and **2,082** words;
   the phrase run's 58,769 / 7,360 are its own unit, and the guard names the
   unit rather than pretending they are the word map's.
3. **Source join.** Every `pair_id` joins the source; the source's counts agree
   (1520 / 524 / 996) and the judged labels agree.
4. **Fold rebuild.** Every judged score and rank percentile reproduce at
   1e-12.
5. **Fill block.** When the abstain run is selected: `floor_pairs == 518`,
   `floor_negative_share` rounds to 0.7780, `non_floor == 1002`.

A failed check blocks the views; the failure is shown, never smoothed over.

## Data-layer API (`tools/omct_error_view.py`)

Pure (numpy + the track's pure cores; no Streamlit, no model client, no
network, no write); deterministic given the files.

- dataclasses: `Sources`, `SourcePair`, `JudgedPair`, `ControlCell`,
  `RunArtifact`, `Provenance`, `GridSet`, `Context`, `Contribution`,
  `SignedWord`, `RankMove`, `PairView`, `ErrorRow`, `QuestionWord`, `QuestionFold`,
  `StabilityLink`, `QuestionView`, `StabilityIndex`, `RunView`, `CompareRow`,
  `Check`, `Reproduction`, `GuardReport`;
- loading: `resolve_sources`, `load_run`, `load_context`,
  `stability_index_from_block`, `load_stability_index`;
- arithmetic: `fold_ids`, `grids_for`, `floored`, `fold_counts`,
  `rank_in_candidates`, `empty_fill`, `signed_words_for`, `contributions_for`,
  `pair_view`,
  `error_of`, `error_rows`, `question_view`, `control_strip`, `run_view`,
  `compare_views`;
- guard: `reproduce`, `check_all`;
- export: `to_csv`; `main(argv)` for `--check` and `--dump`
  (`pairs` / `errors` / `control` / `question`).

`tools/omct_error_app.py` is the only file that imports Streamlit (and pandas
for display). No new dependency: Streamlit is already the UPR viewer's dev
dependency.

## Acceptance checks

1. **Hand-computed unit tests** (`tests/test_omct_error_view.py`, a tiny
   synthetic run; no network/model):
   - the fold scheme is the track's (five contiguous chunks of the seeded
     permutation);
   - a non-floored pair's rebuilt score equals the fold map's positive z-scores
     summed over the paragraph's units **by hand**, and the contributions sum
     to the score;
   - the signed table keeps the negative side (a superset of the positive
     decomposition, whose summed contributions still equal the score), and on
     the real `AI-30-Q276` the word `within` is negative for the z arm
     (`z = -0.148466…`, contribution 0) and positive for the log-odds arm
     (`log-odds = +2.564949…`);
   - the floor / neutral / max fills on `[1, 3, 8]` are `0 / 3 / 8`, and on an
     empty row `-1 / 0 / -1`;
   - the class boundaries (2 / 3 / 52 / 53 of 209) and the toy question's
     classes;
   - the error taxonomy, including the floored class on both labels;
   - the control cells are never in the errors and carry no label;
   - the compare movement is right minus left;
   - the guard is green on the toy, and a doctored map count or a doctored
     score is rejected (the score case also withholds the contributions);
   - the CLI exits 0 then 1;
   - an AST/read test asserts the data layer imports no Streamlit, torch,
     sentence-transformers, httpx or sqlite3.
2. **Real-artifact checks** (`python tools/omct_error_view.py --check`):
   exit 0 on every offered scorer run; the default run's max score and
   rank-percentile differences are **0.0**; the abstain run's fill block reads
   518 / 0.7780 / 1002; the CS-48 map-only run is refused with its message.
   Acceptance anchors: the default run's pair `AI-11-Q70` has the artifact
   score **4.406361730316086** and rank **1 / 51**; the evidence cut gives
   **492** false positives, **10** false negatives, **518** floored pairs
   (**115** floored positives) on the default run.
3. **App smoke test** (Streamlit `AppTest`, no browser, no network; skipped
   when `results/` is absent): the four views render on the real default run
   with no exception and no guard banner.
4. **The gate**: `ruff check` and `ruff format --check` over `src`, `tests`,
   `tools`; the full `pytest` suite; `git status` shows only the intended
   files; nothing under `results/` changed.

## Not in V1

No pack/LLM views, no causal what-if, no export view, no new run, no artifact
rewrite, no log or table edit, no change to the pipeline or to the UPR viewer.
No POS is invented; no control cell is labelled; no number is presented as a
finding that is not the artifact's own or a labelled recomputation. If a real
discrepancy appears, it is reported, not fixed.
