# Quality experiments and maintenance

Run these commands from the repository root as Python modules. Install
`requirements-pipeline.txt` first. They use the existing input data, reports,
checkpoints, and the production quality model at the repository root; moving
the scripts did not move or discard those local files.

| Command | Purpose |
| --- | --- |
| `python -m experiments.preproc_quality_filter_calibration` | Rebuild the quality-filter calibration report. |
| `python -m experiments.quality_distill_label --dry-run` | Prepare and estimate the paid labeling batch. |
| `python -m experiments.quality_distill_label --submit` | Submit the paid labeling batch. |
| `python -m experiments.quality_distill_label --fetch BATCH_ID` | Fetch labels from a completed batch. |
| `python -m experiments.quality_distill_train` | Rebuild the production quality model from local labels. |
| `python -m experiments.quality_delete_dryrun --limit 5000` | Scan a small Pinecone sample and write a report; deletes nothing. |
| `python -m experiments.quality_delete_dryrun --all` | Scan the full index; deletes nothing. |
| `python -m experiments.quality_tag_vectors --dry-run` | Preview vector tagging from the report. |
| `python -m experiments.quality_tag_vectors --tag` | Resume tagging vectors in Pinecone. |
| `python -m experiments.quality_tag_vectors --untag` | Revert tags. |
| `python -m experiments.quality_delete_orphans --verify` | Recheck the historical orphan range. |
| `python -m experiments.quality_delete_orphans --execute` | Delete only verified orphans in that range. |

`quality_filter_candidate.py` stays at the repository root because the live
app and ingestion pipeline import it. `quality_distill_model.json` stays there
because the live app loads it at runtime. File paths are anchored to the
repository root; commands may also be launched elsewhere when the repository
is on `PYTHONPATH`.
