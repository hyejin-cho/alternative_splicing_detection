# SI — Splicing Investigator (Tentative)

A pipeline for detecting alternative splicing (AS) events from RNA-seq BAM files
and filtering out alignment artifacts using a gradient boosting classifier.

> ⚠️ **Work in progress.** This is an actively developed research project and is
> **not a completed or production-ready tool**. Interfaces, file formats, and
> behaviour may change. Expect rough edges, incomplete documentation, and
> features that are still being validated. Use at your own discretion and please
> verify results independently.

---

## Overview

The pipeline has two main stages:

1. **Detection** (`detect_AS.py`) — parses BAM files against a GTF annotation and
   a novel-splice-site file to identify alternative splicing events (ES, A3SS,
   A5SS, IR, MXE), parallelised across samples and chromosomes.
2. **Filtration** (`filter_AS_artifacts.py`) — trains a gradient boosting
   classifier (LightGBM, with XGBoost fallback) to distinguish true splicing
   events from alignment artifacts, then applies it to the detection output.

A supporting set of scripts prepares training data (from simulation and from the
GTEx database), builds a ground-truth set from long-read data, and assembles the
final labelled training set.

---

## Repository structure

```
.
├── detect_AS.py              # Main AS detection entry point
├── filter_AS_artifacts.py    # Gradient boosting artifact filter (train + predict)
├── config.ini                # Paths and output settings
├── classes/                  # Core detection modules
│   ├── file.py               #   Base file handling
│   ├── gtf.py                #   GTF parsing + gene interval index
│   ├── bam.py                #   BAM parsing (pysam) + junction counting
│   ├── junctions.py          #   Junction index + novel-splice classification
│   ├── events.py             #   AS event formation + output writing
│   └── splice_site_features.py  # Splice site feature annotation (donor/acceptor,
│                                #   MaxEntScan, annotation distances)
├── scripts/                  # Data preparation and benchmarking utilities
│   ├── combine_novel_splices_SJ.pl   # Combine STAR SJ.out.tab novel junctions
│   ├── extract_true_junctions.py     # True junctions from simulation GTF
│   ├── label_gtex_junctions.py       # True junctions from GTEx (streaming)
│   ├── generate_negatives.py         # Negative-label (artifact) junctions
│   ├── assign_labels.py              # Label detected junctions vs ground truth
│   ├── build_training_set.py         # Merge positives + negatives → training set
│   ├── build_ground_truth.py         # Long-read ground truth (PacBio/ONT)
│   ├── check_bam_compatibility.py    # Check long-read BAMs before merging
│   ├── sashimi-plot.py               # Sashimi plots for visualisation
│   └── make_plots4AS.pl              # Plot job generation
└── stats/
    ├── calcDE.py             # Differential AS calculation
    └── diff_exp.R            # R differential expression analysis
```

*(Some paths above reflect the intended layout; adjust to match your actual
checkout.)*

---

## Requirements

- **Python** 3.6+ (developed and tested primarily on 3.6)
- **Perl** 5 (for the `.pl` helper scripts)
- **R** (for differential analysis)
- **External tools:** samtools, STAR (for alignment), and optionally minimap2
  (for long-read ground truth)

### Python packages

```bash
pip install pysam pandas numpy pyranges scikit-learn lightgbm shap joblib
# maxentpy is optional (for MaxEntScan splice site scores)
pip install maxentpy
```

---

## Quick start

### 1. Detect AS events

```bash
python detect_AS.py \
    -b1 s1.bam,s2.bam \
    -b2 s3.bam,s4.bam \
    -g  annotation.gtf \
    -s  combined_novel_splices.txt \
    -d  output_dir \
    -o  results \
    --fasta genome.fa        # optional: adds splice site feature columns
```

Output files are written as
`output_dir/results_<group>_<AS_TYPE>_<sample>_events.txt`.

> **Note:** BAM files must be coordinate-sorted and indexed (`samtools index`),
> and chromosome naming in the BAM, GTF, novel-splice file, and FASTA must all
> match (e.g. all `chr1` or all `1`). Mismatched naming silently yields empty
> results.

### 2. Filter artifacts

```bash
python filter_AS_artifacts.py \
    --training    final_training_set.tsv \
    --events-dir  output_dir/ \
    --output-dir  filtered_output/ \
    --model-out   artifact_filter.pkl
```

---

## Preparing training data

The filter needs a labelled training set. The supporting scripts produce one:

```bash
# Positives from simulation
python scripts/extract_true_junctions.py --gtf annotation.gtf -o true_junctions.tsv

# Positives from GTEx (streaming — handles the full junction matrix)
python scripts/label_gtex_junctions.py --gtex GTEx_junctions.gct.gz \
    --fasta genome.fa -o gtex_true_junctions.tsv

# Negatives (artifacts)
python scripts/generate_negatives.py --labeled labeled_junctions.tsv \
    -o negatives.tsv

# Merge into the final training set
python scripts/build_training_set.py \
    --sim-pos  true_junctions.tsv \
    --gtex-pos gtex_true_junctions.tsv \
    --neg      negatives.tsv \
    --max-ratio 3 \
    -o final_training_set.tsv
```

See each script's `--help` for full options.

---

## Method summary

- **Detection** counts junction-spanning and exonic reads per candidate event,
  computes PSI, and classifies events by type (ES / A3SS / A5SS / IR / MXE).
- **Filtration** frames artifact removal as binary classification on tabular
  features (read counts, PSI, splice site scores, annotation distances). It uses
  a **LightGBM** gradient boosting classifier, evaluated by **PR-AUC** under
  stratified cross-validation, with **SHAP** values for feature interpretability.

---

## Known limitations & caveats

- This is a **research prototype**, not a validated release. Results have not
  been comprehensively benchmarked across datasets.
- Chromosome-naming consistency between inputs is required and not yet
  auto-normalised everywhere.
- The training data preparation involves several manual steps and choices
  (filter thresholds, class ratios) that affect the final model.
- Some scripts assume specific file layouts and may need path edits.
- Test coverage is minimal.

---

## Status & roadmap

Currently working toward:

- More robust input validation (chromosome naming, coordinate conventions)
- Benchmarking against established tools (rMATS, SUPPA2, MAJIQ, LeafCutter)
- Long-read validation of filtered results
- Packaging and documentation improvements

Contributions, issues, and suggestions are welcome, but please keep in mind the
early-stage nature of the project.

---

## Citation

A manuscript is in preparation. If you use this code in the meantime, please
link back to this repository.

---

## License
No license has been specified yet. Until one is added, default copyright applies
and reuse rights are limited — please contact the author before reusing.

---

## Contact

Maintained by the repository owner. Please open a GitHub issue for questions or
bug reports.
