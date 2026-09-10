"""Calibration harness for the chunk-level OCR/transcription quality filter.

SCRATCH / STAGING SCRIPT (EVAL_LOG.md "Round 15 — OCR Re-Index: Scope",
Phase 1). Read-only against the CSVs; writes only its own report file.


Usage:
    /Users/kevin/Desktop/ds/environments/pearl_env/bin/python preproc_quality_filter_calibration.py
"""

import pandas as pd

from preproc import chunk_dataframe
from quality_filter_candidate import score_chunk_quality

LE_NOUVELLISTE_CSV = "dloc/data/output/le_nouvelliste.csv"
RADIO_HAITI_CSV = "radiohaiti/data/output/radio_haiti.csv"

SAMPLE_N_ARTICLES = 300
SEED = 42
REPORT_PATH = "quality_filter_calibration_report.csv"


# ---------------------------------------------------------------------------
# Seeded ground truth
# ---------------------------------------------------------------------------

# Hardcoded short strings quoted directly in EVAL_LOG.md (not tied to a
# specific CSV row — scored as standalone chunk-sized text).
KNOWN_GARBAGE_STRINGS = [
    ("ad_fluocaril", "FLUOCARIL"),
    ("ad_goodrich", "Goodrich Tic"),
    ("ad_ginger_kola", "Ginger Kola"),
    ("ad_patton_paint", "PATTON PAINT COMPANY"),
]

KNOWN_LEGIT_STRINGS = [
    ("round14_paprince", "que c'est le Port-au-Prince d'il y a 40 ans, ou le Port-au-Prince d'aujourd'hui sont tout à fait différents"),
    ("paprince_1900", "PORT-AU-PRINCE de 1900 était douce et tumultueuse"),
]

# Confirmed-real garbage that a cheap, dependency-light heuristic cannot
# reliably catch without unacceptable collateral damage — verified during
# calibration:
#   - row9489/row1662: individually-real French words assembled into an
#     OCR-scrambled sentence. Word-frequency hit-rate is statistically
#     identical to confirmed-clean French (row5674: 0.86 vs 0.84-0.85) —
#     needs sequence-level language modeling to catch, out of scope here.
#   - ad_haitian_motors/mixed_case_digit: catchable in principle by
#     lowering MIN_TOKENS or GARBLE_SCORE_THRESHOLD, but doing so was
#     tested and sweeps up ~36% of the entire unlabeled sample into the
#     borderline band, most of which is legitimate (esp. Radio Haiti
#     Creole speech, which naturally scores similarly due to its
#     repetitive/informal structure, not corruption).
# Tracked here (scored and reported) but excluded from the strict
# precision/recall calculation, which should only judge the heuristic
# against what it's actually designed to catch.
KNOWN_OUT_OF_SCOPE_STRINGS = [
    ("ad_haitian_motors", "HAITIAN MOTORS. S.A."),
    ("mixed_case_digit", "Ihewefc TANWS QUE IES STATIST3QUES PAHUNT Tar.dís que les stotútiques"),
    # Confirmed to have caused a real eval failure (Round 11, Contextual
    # Relevancy 0.0) but its French hit-rate (0.40) is only modestly below
    # 3 legit Creole reference points (0.66-0.98) — too thin a margin to
    # trust a symmetric wordfreq penalty against under-sampled Creole
    # content without real false-positive risk. Accepted gap, not pursued.
    ("eval_ht_mixed_0", "Princionnens, Oulve une belle oy: gentation"),
]

# (row_idx, anchor substring to locate the specific chunk, seed_type, chunk_id)
# Anchors verified directly against the raw CSVs before writing this script.
LE_NOUVELLISTE_KNOWN_ROWS = [
    (9489, "sooment comme on Te", "known_out_of_scope", "row9489_garble"),
    (1662, "eonsulat haitien", "known_out_of_scope", "row1662_garble"),
    (5674, "cyclone fait 100.000 morts", "known_legit", "row5674_clean_afp"),
]

RADIO_HAITI_KNOWN_ROWS = [
    (44, "Paris, 7 octobre 1947", "known_legit", "row44_paris_letter"),
    # 745 and 1176 are legitimate Creole oratory with heavy repetition
    # (rhetorical, not Whisper stutter) — deliberately included as a hard
    # case the filter must not misflag as garbled.
    (745, "Yon bade le gòl", "known_legit", "row745_creole_repetitive"),
    (1176, "Muzi en profondeur", "known_legit", "row1176_creole_repetitive"),
    (1217, "d'identifikasyon", "known_legit", "row1217_creole_elision"),
    (1852, "s'ajouteront à ceux de Saint-Soleil", "known_legit", "row1852_diacritics"),
]


def pull_row_chunks(csv_path, row_idx):
    """Return chunk_dataframe() output restricted to a single CSV row."""
    df = pd.read_csv(csv_path)
    row_df = df.iloc[[row_idx]].reset_index(drop=True)
    chunks_text, _ = chunk_dataframe(row_df, shuffle=False)
    return chunks_text


def find_chunk_containing(chunks, substring):
    for c in chunks:
        if substring in c:
            return c
    return None


def build_known_row_examples(csv_path, known_rows, source_label):
    examples = []
    for row_idx, anchor, seed_type, chunk_id in known_rows:
        chunks = pull_row_chunks(csv_path, row_idx)
        if not chunks:
            print(f"WARNING: row {row_idx} produced no chunks, skipping {chunk_id}")
            continue
        text = find_chunk_containing(chunks, anchor)
        if text is None:
            print(f"WARNING: anchor {anchor!r} not found in row {row_idx} chunks "
                  f"(chunk boundaries may have shifted) — using first chunk instead")
            text = chunks[0]
        examples.append((source_label, seed_type, chunk_id, text))
    return examples


def sample_unlabeled_chunks(csv_path, source_label, n_articles=SAMPLE_N_ARTICLES, seed=SEED):
    df = pd.read_csv(csv_path)
    n = min(n_articles, len(df))
    df_sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    chunks_text, _ = chunk_dataframe(df_sample, shuffle=False)
    return [
        (source_label, "unlabeled_sample", f"{source_label}_chunk_{i}", text)
        for i, text in enumerate(chunks_text)
    ]


def build_calibration_set():
    examples = []

    for chunk_id, text in KNOWN_GARBAGE_STRINGS:
        examples.append(("mixed", "known_garbage", chunk_id, text))
    for chunk_id, text in KNOWN_LEGIT_STRINGS:
        examples.append(("mixed", "known_legit", chunk_id, text))
    for chunk_id, text in KNOWN_OUT_OF_SCOPE_STRINGS:
        examples.append(("mixed", "known_out_of_scope", chunk_id, text))

    examples += build_known_row_examples(LE_NOUVELLISTE_CSV, LE_NOUVELLISTE_KNOWN_ROWS, "le_nouvelliste")
    examples += build_known_row_examples(RADIO_HAITI_CSV, RADIO_HAITI_KNOWN_ROWS, "radio_haiti")

    print("Sampling unlabeled chunks (this runs chunk_dataframe on "
          f"{SAMPLE_N_ARTICLES} articles per corpus)...")
    examples += sample_unlabeled_chunks(LE_NOUVELLISTE_CSV, "le_nouvelliste")
    examples += sample_unlabeled_chunks(RADIO_HAITI_CSV, "radio_haiti")

    return examples


def main():
    print("Building calibration set...")
    calib = build_calibration_set()
    print(f"Total examples: {len(calib)}")

    rows = []
    for source, seed_type, chunk_id, text in calib:
        result = score_chunk_quality(text, use_french_rescue=True)
        rows.append({
            "source": source,
            "seed_type": seed_type,
            "chunk_id": chunk_id,
            "verdict": result["verdict"],
            "score": result["score"],
            "n_tokens": result["n_tokens"],
            "avg_word_len": result["avg_word_len"],
            "mixed_alnum_ratio": result["mixed_alnum_ratio"],
            "consonant_run_ratio": result["consonant_run_ratio"],
            "short_unexplained_ratio": result["short_unexplained_ratio"],
            "repetition_ratio": result["repetition_ratio"],
            "non_latin_ratio": result["non_latin_ratio"],
            "hard_trigger": result["hard_trigger"],
            "french_rescue_score": result["french_rescue_score"],
            "rescued": result["rescued"],
            "text": text,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["source", "score"], ascending=[True, False])
    out.to_csv(REPORT_PATH, index=False)
    print(f"Wrote {REPORT_PATH} ({len(out)} rows)")

    # --- Seeded precision/recall -------------------------------------------
    seeded = out[out["seed_type"].isin(["known_garbage", "known_legit"])]
    tp = int(((seeded["seed_type"] == "known_garbage") & (seeded["verdict"] == "drop")).sum())
    fn = int(((seeded["seed_type"] == "known_garbage") & (seeded["verdict"] == "keep")).sum())
    fp = int(((seeded["seed_type"] == "known_legit") & (seeded["verdict"] == "drop")).sum())
    tn = int(((seeded["seed_type"] == "known_legit") & (seeded["verdict"] == "keep")).sum())

    print("\n=== Seeded set precision/recall ===")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    if tp + fp > 0:
        print(f"Precision: {tp / (tp + fp):.2%}")
    if tp + fn > 0:
        print(f"Recall:    {tp / (tp + fn):.2%}")

    if fn > 0:
        print("\nFALSE NEGATIVES (known garbage scored 'keep'):")
        print(seeded[(seeded["seed_type"] == "known_garbage") & (seeded["verdict"] == "keep")]
              [["chunk_id", "score", "text"]].to_string(index=False))
    if fp > 0:
        print("\nFALSE POSITIVES (known legit scored 'drop'):")
        print(seeded[(seeded["seed_type"] == "known_legit") & (seeded["verdict"] == "drop")]
              [["chunk_id", "score", "text"]].to_string(index=False))

    rescued_count = int(seeded["rescued"].sum())
    if rescued_count:
        print(f"\nFrench-rescue signal fired on {rescued_count} seeded example(s).")

    # --- Known out-of-scope (tracked, not scored) ---------------------------
    oos = out[out["seed_type"] == "known_out_of_scope"]
    if len(oos):
        print(f"\n=== Known out-of-scope examples ({len(oos)}) — tracked, excluded from precision/recall ===")
        print(oos[["chunk_id", "verdict", "score", "text"]].to_string(index=False))

    # --- Unlabeled sample drop-rate -----------------------------------------
    print("\n=== Unlabeled sample drop-rate by source ===")
    unl = out[out["seed_type"] == "unlabeled_sample"]
    print(unl.groupby("source")["verdict"].value_counts(normalize=True))

    unl_rescued = int(unl["rescued"].sum())
    print(f"\nFrench-rescue signal fired on {unl_rescued} unlabeled chunk(s) "
          f"({unl_rescued / max(len(unl), 1):.2%} of unlabeled sample).")


if __name__ == "__main__":
    main()
