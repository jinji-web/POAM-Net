# POAM-Net — PyTorch Code and Experimental Pipeline

**Paper:** *Public Opinion-Aware Multimodal Learning via Cross-Modal and Content–Opinion Discrepancy Modeling for Fake News Detection*  
**Author:** Jin Ji  

This repository provides the POAM-Net PyTorch model, data-processing utilities, training and evaluation pipelines, ablation configurations, robustness analyses, statistical evaluation and visualization tools. Results are computed from the user-provided datasets and trained model checkpoints; numerical results from the paper are not embedded as evaluation outputs.

## 1. What is implemented

The implementation follows the detailed POAM-Net method in Sections 3.1–3.8 of the manuscript:

| Manuscript | Implementation |
|---|---|
| §3.3 Text features (XLM-RoBERTa-base) | Shared `text_encoder`, `text_project`, `comment_project` |
| §3.3 Visual features (Swin Transformer Tiny) | `image_encoder`, patch and pooled projections |
| §3.4 Bidirectional text-to-image / image-to-text attention | Two multihead attention layers (`heads=8`) |
| Eq. (17–20): text–image discrepancy | Absolute difference, elementwise product, cosine + MLP |
| Eq. (21–28): comment reliability aggregation | Relevance, agreement, masked attention, learnable null opinion |
| Eq. (29–32): content–opinion discrepancy | Projected feature pairs, absolute difference, product, cosine + MLP |
| Eq. (33–38): four-way adaptive fusion | Softmax gate, weighted fusion, residual FFN, sigmoid classifier |
| Eq. (39–41): weighted BCE / regularization | `BCEWithLogitsLoss`, optional class weighting, AdamW decay |

Supporting experiments include: training, held-out testing, metrics, per-post saved predictions, repeat seeds, controlled baselines, five main ablations, published-data preparation adapters, a fixed-cohort temporal analysis, split overlap diagnostics, comment perturbations, missingness stress tests, transfer fine-tuning, paired confidence intervals, and plots from actual model predictions.

**Model and configuration details:** Architecture settings, tensor conventions, MLP widths, normalization, dropout, attention scoring and optimization settings are defined in `poam/model.py` and the configuration YAML files. Match the dataset version, split manifests, checkpoints, seeds and configuration used for each reported experiment when regenerating numerical tables. This repository includes POAM-Net, ablations and matched-input local controls. SpotFake, SAFE, HMCAN, CAFE, SAMPLE and SARD require their separately obtained implementations to reproduce the full external-baseline comparison in Table 3.

## 2. Installation

Use Python 3.10–3.12 and a PyTorch-compatible NVIDIA GPU for full-size training. The paper used a large training set and a shared transformer applied to as many as 20 comments per post; its full settings can require substantial GPU memory, so the paper's `batch_size=16` may need smaller local batches plus gradient accumulation. Local throughput depends on hardware.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Pretrained encoders require Hugging Face access or locally cached compatible checkpoints. `train.py` downloads them using `transformers.AutoModel.from_pretrained(...)` the first time. This package does not ship large model weights. If loading is restricted, set `text_checkpoint` and `image_checkpoint` to authorized local checkpoint paths. There is no included dataset download because the benchmark access conditions and raw versions must be respected.

## 3. Required data layout

Run commands **from the repository root**. The standardized data format is one JSON object per line; the image path is resolved relative to `image_root` in `configs/*.yaml`.

```
POAM_Net_Code/
  data/
    fakeddit/
      images/...
      train.jsonl
      val.jsonl
      test.jsonl
    weibo/
      images/...
      train.jsonl
      val.jsonl
      test.jsonl
```

Each JSONL row has this shape (see `examples/record_schema.json`):

```json
{"id":"unique-post-id", "text":"original post text", "image":"image_001.jpg", "label":0,
 "post_ts":"2025-01-01T12:00:00+00:00",
 "comments":[{"text":"actual response","created_ts":"2025-01-01T12:30:00+00:00"}]}
```

**Critical label definition:** `0=REAL; 1=FAKE` (manuscript Eq. 3). Confirm the source's binary label convention; do not assume Fakeddit raw `2_way_label` shares this convention without checking. Map from the verified raw labels using `--label-map`. An empty comment array `[]` is legitimate. Timestamps must have a timezone (`Z` is accepted) or be Unix seconds; leave absent where genuinely missing. Do not infer timestamps. **Do not put a repost or duplicate of a held-out post in training.**

### Fakeddit adapter

The manuscript uses the official multimodal-only split with 544,288 training, 57,551 validation, and 57,525 test instances. Download the data from its original provider according to its licence, and map the actual tabular fields from your release. Example, **only if these headers exist in your files**:

```bash
python -m scripts.prepare_dataset \
  --train raw/multimodal_only_train.tsv \
  --val raw/multimodal_only_validate.tsv \
  --test raw/multimodal_only_test.tsv \
  --comments raw/comments.csv \
  --output data/fakeddit \
  --id-column id --text-column clean_title \
  --image-column image_path --label-column 2_way_label \
  --comment-id-column post_id --comment-text-column body \
  --comment-time-column created_utc \
  --post-time-column created_utc --label-map 0:0,1:1
```

The `0:0,1:1` mapping in the example is **only a syntax illustration**; change it if your source's label meanings differ. The `image_path` field, image filenames and comment export may need a small local adapter. If comments are nested within each post already, use the standardized schema directly instead of the adapter. If you omit `--comments`, the adapter writes empty lists; this cannot reproduce comment-aware experiments.

### SocialNet-Weibo V2 adapter

The manuscript reports 11,329 posts (5,661 fake, 5,668 real), with **70/10/20 stratified split** seeded at 42. Point `--all` to a structured CSV/TSV/JSONL export with real texts, image paths and verified labels. Example:

```bash
python -m scripts.prepare_dataset \
  --all raw/weibo_posts.csv --comments raw/weibo_comments.csv \
  --output data/weibo \
  --id-column post_id --text-column text \
  --image-column image_file --label-column authenticity \
  --label-map real:0,fake:1
```

The script stratifies training/validation/test as specified, without sharing IDs. Again, field names depend on the actual downloaded release.

## 4. Main model training, validation, test

The paper's Table 2 settings are in `configs/fakeddit.yaml` and `configs/weibo.yaml`: news 256 tokens, comments 64, maximum comments 20, images 224×224, shared dimension 256, attention heads 8, dropout 0.20, batch size 16, `encoder_lr=2e-5`, `head_lr=1e-4`, AdamW weight decay 0.01, 15 epochs, patience 3, gradient clipping 1, warmup 0.10, threshold 0.5.

```bash
python -m scripts.train --config configs/fakeddit.yaml --output runs/fakeddit/full/42 --seed 42
python -m scripts.evaluate --checkpoint runs/fakeddit/full/42/best.pt --output runs/fakeddit/full/42/test
python -m scripts.plot_results --predictions runs/fakeddit/full/42/test/predictions.jsonl --output runs/fakeddit/full/42/figures
```

For limited VRAM, adjust *effective* batch size deliberately:

```bash
python -m scripts.train --config configs/fakeddit.yaml \
    --output runs/fakeddit/full/42 --batch-size 2 --grad-accumulation 8
```

The optimizer takes a smaller learning rate on pretrained encoders; training uses validation Macro-F1 for checkpoint selection, **never the test set**. `best.pt` contains weights, seed, selected epoch, YAML configuration and validation metrics. `predictions.jsonl` holds post IDs, labels, probabilities, counts and available attention/gate diagnostics; `metrics.json` contains Accuracy, Macro-F1, class metrics, AUC, Brier, ECE and confusion matrix.

## 5. Five-seed repetitions, baseline controls, ablation

```bash
python -m scripts.experiment_matrix \
  --config configs/weibo.yaml --output runs/weibo/study \
  --seeds 42 43 44 45 46 \
  --variants full no_tid no_ra no_cod no_af no_opinion \
             comment_mean comment_attention concat_mlp gated_fusion \
             product_only relevance_only
```

Variants correspond to Table 4 (`no_tid`, `no_ra`, `no_cod`, `no_af`, `no_opinion`), comment-aware controls (mean and attention), and Table 11-style architecture controls (`concat_mlp`, `gated_fusion`, `product_only`, `relevance_only`). `text_only` and `image_only` are also available. These variants provide the model configurations used for the corresponding local control experiments. Layer count and parameter matching should be checked from measured model statistics. Read actual parameter counts from `training_summary.json`; the manuscript itself does not give parameter counts. `experiment_matrix.py` calculates seed means and standard deviations from the completed training runs.

## 6. Strict publication-relative comment windows

If valid post times are missing, do **not** construct a temporal benchmark by guessing timestamps. Produce a fixed population first, and compare the *same posts* at each horizon:

```bash
python -m scripts.temporal_cohort \
   --input data/weibo/test.jsonl --output data/weibo/timestamp_valid_test.jsonl
python -m scripts.temporal_eval \
   --checkpoint runs/weibo/full/42/best.pt \
   --cohort data/weibo/timestamp_valid_test.jsonl \
   --output runs/weibo/temporal --horizons 1 6 24
```

Undated comments are excluded from the bounded horizons. In the all-available condition, the original comment archive is used. This distinction is intentional: a time-censored comment set and retrospective full archive are different information conditions. `temporal_cohort.py` records the eligible post count and percentage. Its output is required for a consistent time-window denominator.

## 7. Irrelevant and coordinated comment stress tests

Prepare a UTF-8 file with one **real independently curated** adversarial or off-topic comment text per line (no target labels inside). The scripts apply deterministic per-post changes, preserve original labels, and record the attack parameters. The attack corpus must be curated and documented separately; the script does not generate externally verified claims or a real bot network.

```bash
python -m scripts.attack_comments \
  --input data/weibo/test.jsonl --output data/weibo/test_attacked.jsonl \
  --attack-corpus inputs/relevant_wrong_comments.txt --fraction 0.2 \
  --kind on_topic --seed 42
python -m scripts.evaluate --checkpoint runs/weibo/full/42/best.pt \
  --path data/weibo/test_attacked.jsonl --output runs/weibo/attack_on_topic
```

For the `repeated_consensus` or `coordinated` modes the code repeats a selected corpus response, so it tests **repetition at a fixed comment budget**; real-world coordinated-bot conclusions require independently documented attack provenance and timestamps. Under an attack, compare the same posts' IDs and probability outputs to obtain prediction-flip rates; do not compare experiments that used different test populations.

## 8. Data audit, missingness and paired statistical inference

```bash
python -m scripts.audit_splits \
  --train data/weibo/train.jsonl --val data/weibo/val.jsonl \
  --test data/weibo/test.jsonl --image-root data/weibo/images \
  --output runs/weibo/split_audit.json
python -m scripts.missingness --input data/weibo/test.jsonl \
  --output data/weibo/test_no_comments.jsonl --mode remove_comments
python -m scripts.evaluate --checkpoint runs/weibo/full/42/best.pt \
  --path data/weibo/test_no_comments.jsonl --output runs/weibo/no_comments
python -m scripts.statistics \
  --poam runs/weibo/full/42/test/predictions.jsonl \
  --baseline runs/weibo/study/comment_attention/42/test/predictions.jsonl \
  --output runs/weibo/paired_post_ci.json --n-bootstrap 2000
```

The split audit checks **exact normalized text SHA-256 hashes** and **identical 64-bit pHashes** across partitions, in addition to overlapping post IDs. Event-disjoint evaluation and broader semantic near-duplicate screening require suitable metadata and additional checks.

The statistical script uses paired **post-level** bootstrap by default, and event-cluster bootstrap only when given a verified ID→event JSON map. An interval from repeated training seeds is a different estimand; compute one using `poam.stats.paired_seed_ci(...)` with matched seed-level differences. For reporting Table 13, record the exact resampling units, confidence-interval method, number of iterations and seed pairing.

For controlled image removal, use `scripts.missingness --mode remove_image` and then `python -m scripts.evaluate ... --allow-missing-images`. The resulting diagnostic is test-time masking only, not a learned missing-image robust model.

## 9. Cross-platform supervised transfer

Run the full source-domain training first. Then fine-tune on fractions of the *target training set*, holding its validation/test files fixed:

```bash
python -m scripts.transfer \
  --target-config configs/weibo.yaml \
  --source-checkpoint runs/fakeddit/full/42/best.pt \
  --target-fractions .01 .05 .10 .25 1.0 \
  --output runs/fakeddit_to_weibo --seed 42
```

Remove `--source-checkpoint` for a scratch target baseline. To evaluate Weibo→Fakeddit, swap configs/checkpoints. This is **target-labelled fine-tuning**, not zero-shot transfer. Regenerate the corresponding transfer tables using the recorded checkpoints, seeds, target split and evaluated model outputs.

### Other included diagnostics

- `python -m scripts.shuffle_comments --input data/weibo/test.jsonl --output data/weibo/shuffled_test.jsonl` permutes held-out comments **within test only**, label-blind, to test correspondence dependence (Table 11-style).
- `python -m scripts.subgroup_eval --predictions <test/predictions.jsonl> --records data/weibo/test.jsonl --output runs/weibo/subgroups.json` computes natural 0/1–2/3–5/6–10/>10 comment strata with actual sample counts (Table 12-style). Short-text and similarity thresholds can be supplied only when fixed in advance from training.
- `python -m scripts.prediction_shift --clean <clean/predictions.jsonl> --attacked <attacked/predictions.jsonl> --output runs/weibo/shift.json` calculates paired prediction-flip rates.
- `python -m scripts.plot_embeddings --checkpoint <best.pt> --output runs/weibo/tsne.png` produces a t-SNE view from actual held-out representations (the output depends on the selected checkpoint, data cohort and random seed).

## 10. Software tests and preparing an archival release

```bash
python -m pytest -q
python -m compileall -q poam scripts tests
```

Unit tests use lightweight stub encoders to check the software interfaces and tensor operations. Training on Fakeddit or SocialNet-Weibo V2 requires the corresponding data, pretrained model weights and a suitable GPU environment.

Before archiving the repository alongside the paper:

1. Restore/verify your exact dataset export, comment joins, raw-to-binary label mapping, sampling, split and feature normalization.
2. Check that code, model configuration and saved checkpoint match the experiments described in the manuscript.
3. Train with recorded seeds, archive run configs and held-out predictions, regenerate all figure/table values using the real runs. Supply external-baseline source/licences where applicable.
4. Document Python/PyTorch/Transformers/CUDA versions, GPU model, deterministic choices, true parameter counts, and the CI resampling units.
5. Publish the confirmed research release on GitHub, archive a release in Zenodo and cite the DOI that Zenodo assigns.

The training, evaluation and analysis commands produce outputs from the configured datasets and checkpoints. Preserve run configurations, prediction files and experiment logs for reproducible reporting.
