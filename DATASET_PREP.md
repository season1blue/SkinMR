# MM-Skin Data Preparation Guide

This document explains how to prepare the datasets for zero-shot classification in this repository, with data stored locally under this project.

## 1. Target Layout

Use this layout under the project root:

    MM-Skin/
      data/
        Patch16/
          ...jpg
        HAM10K/
          test/images/ISIC2018_Task3_Test_Input/
            ISIC_XXXXXXX.jpg
        PAD/
          images/
            imgs_part_1/
            imgs_part_2/
            imgs_part_3/

The evaluation CSV files are expected at:

    Dataframe/test/classification/Patch16_2class_test_10pct_seed42.csv
    Dataframe/test/classification/HAM10K_ISIC2018_test.csv
    Dataframe/test/classification/PAD_test.csv

## 2. Download Links (Official Sources)

- Patch16: https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S
- HAM10000 / ISIC 2018: https://challenge.isic-archive.com/data/#2018
- PAD (dataset page): https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S
- PAD (direct data.zip): https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/DATA/7QCR8S/LNW2GV

Notes:
- These datasets may require manual registration, approval, or paper/data-request steps.
- If direct scripted download is unavailable, download manually and then move files into the required directories above.

## 3. Place Files Into This Repo

From MM-Skin root, create directories first:

    mkdir -p data/Patch16
    mkdir -p data/HAM10K/test/images/ISIC2018_Task3_Test_Input
    mkdir -p data/PAD/images

Then copy your downloaded images into these folders so that CSV image paths resolve correctly.

## 4. Validate Dataset Integrity Before Running

Run this check script from MM-Skin root:

    python - <<'PY'
    import csv
    import os

    checks = [
        "Dataframe/test/classification/Patch16_2class_test_10pct_seed42.csv",
        "Dataframe/test/classification/HAM10K_ISIC2018_test.csv",
        "Dataframe/test/classification/PAD_test.csv",
    ]

    missing_total = 0
    for csv_path in checks:
        miss = 0
        total = 0
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                p = row["image"]
                if not os.path.exists(p):
                    miss += 1
        missing_total += miss
        print(f"{csv_path}: total={total}, missing={miss}")

    if missing_total == 0:
        print("All required images exist.")
    else:
        print("Some images are missing. Fix paths/files before evaluation.")
    PY

If missing is not zero, check whether:
- data directory names are exactly Patch16, HAM10K, PAD
- nested subdirectories match the CSV image paths
- filenames keep original case and extension

## 5. Quick Run Example (qwen25vl + memvr)

In ZS_classify_test.sh, set:

    LLM_KEY="qwen25vl"
    METHOD="memvr"

Then run one dataset at a time by switching DATASET_KEY:

    DATASET_KEY="patch16_2class"
    bash ZS_classify_test.sh

    DATASET_KEY="ham10k"
    bash ZS_classify_test.sh

    DATASET_KEY="pad"
    bash ZS_classify_test.sh

## 6. Common Pitfalls

- Deleting Dataframe folder will break evaluation immediately.
- Wrong data root path will cause large missing counts.
- Patch16 default is subset mode in ZS_classify_test.sh when PATCH16_USE_SUBSET=1.
