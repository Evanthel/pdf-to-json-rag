# Kaggle publishing bundle

This directory builds a small public dataset and an executable notebook from the project's tracked benchmark snapshots. It deliberately excludes source PDFs, extracted document text, model weights, credentials, and user uploads.

## Build and verify locally

From the repository root:

```bash
python deploy/kaggle/build_assets.py
MPLCONFIGDIR=/tmp/pdf-to-json-rag-matplotlib python deploy/kaggle/build_assets.py --verify
```

The build is deterministic. Use `--check` in automation when the source snapshots change.

## Generated packages

- `dataset/` contains Kaggle dataset metadata, two source snapshots, analysis-ready CSV projections, the project license, and a cover image.
- `notebook/` contains the executable case study and Kaggle kernel metadata.

The notebook runs without a GPU, network access, package installation, API keys, or the original PDFs. It looks for the attached Kaggle dataset first and also supports execution from the repository root.

## Publish safely

Install the official Kaggle CLI in a dedicated environment and authenticate outside this repository. Never copy `kaggle.json` into the project.

```bash
python -m venv /tmp/pdf-to-json-rag-kaggle-venv
source /tmp/pdf-to-json-rag-kaggle-venv/bin/activate
python -m pip install kaggle
kaggle auth login
kaggle datasets create -p deploy/kaggle/dataset
```

The dataset starts private because the create command omits `--public`. Review it and make it public in Kaggle's web interface. Then publish the notebook, which is configured as public and runs against that dataset:

```bash
kaggle kernels push -p deploy/kaggle/notebook
```

Confirm that the notebook run completes successfully before sharing it.

Expected resource IDs:

- Dataset: `piotrobiegly/pdf-to-json-rag-benchmark-results`
- Notebook: `piotrobiegly/pdf-to-json-rag-retrieval-benchmark`

If the account slug differs, update `DATASET_ID` and `NOTEBOOK_ID` in `build_assets.py`, rebuild, and verify before publishing.

Official references: [Kaggle CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication), [dataset metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md), and [notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
