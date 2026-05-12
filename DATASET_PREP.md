# SkinMR Data Preparation Guide

This guide documents the verified data layout and commands used on this machine.

## 1. Canonical Data Root

Use the shared data root:

        /mnt/data/ssz/Skin/skindata

Expected layout:

        /mnt/data/ssz/Skin/skindata/
            Patch16/
            HAM10K/test/images/ISIC2018_Task3_Test_Input/
            PAD/images/
                imgs_part_1/
                imgs_part_2/
                imgs_part_3/

Compatibility link for old scripts:

        /mnt/data/ssz/Skin/SkinMR/data -> ../skindata

## 2. Official Dataset Sources

- Patch16 (Heidata): https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S
- HAM10000 / ISIC 2018 (official challenge page): https://challenge.isic-archive.com/data/#2018
- PAD-UFES-20 (Mendeley dataset page): https://data.mendeley.com/datasets/zr7vgbcyr2/1

## 3. PAD Download (Mendeley, With gg Prefix)

The following command set is verified and uses gg before wget.

        cd /mnt/data/ssz
        mkdir -p Skin/skindata/PAD/raw Skin/skindata/PAD/images

        cat > /tmp/pad_mendeley_files.txt <<'EOF'
        imgs_part_1.zip|1245184680|0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/1cc2f71f-20a2-412d-b746-a9b9bc20c966/file_downloaded
        imgs_part_2.zip|1126646990|e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/559a60ed-5504-475d-996c-6a8bc253b5e7/file_downloaded
        imgs_part_3.zip|1220565093|ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/34dcdf8e-e5f1-4b35-aa0b-5135051ff852/file_downloaded
        EOF

        while IFS='|' read -r name expect_size expect_sha url; do
            out="Skin/skindata/PAD/raw/$name"
            tmp="${out}.part"
            rm -f "$tmp"
            gg wget --tries=0 --retry-connrefused --waitretry=5 --read-timeout=30 --timeout=30 -O "$tmp" "$url"
            [ "$(stat -c%s "$tmp")" -eq "$expect_size" ]
            echo "$expect_sha  $tmp" | sha256sum -c -
            mv -f "$tmp" "$out"
        done < /tmp/pad_mendeley_files.txt

        for z in Skin/skindata/PAD/raw/imgs_part_1.zip Skin/skindata/PAD/raw/imgs_part_2.zip Skin/skindata/PAD/raw/imgs_part_3.zip; do
            unzip -q -o "$z" -d Skin/skindata/PAD/images
        done

## 4. Validation Against CSV Files

Run from:

        cd /mnt/data/ssz/Skin/SkinMR

Then execute:

        python - <<'PY'
        import os, csv
        checks=[
            ('Patch16','Dataframe/test/classification/Patch16_2class_test_10pct_seed42.csv','/mnt/data/ssz/Skin/skindata/Patch16'),
            ('HAM10K','Dataframe/test/classification/HAM10K_ISIC2018_test.csv','/mnt/data/ssz/Skin/skindata/HAM10K/test/images/ISIC2018_Task3_Test_Input'),
            ('PAD','Dataframe/test/classification/PAD_test.csv','/mnt/data/ssz/Skin/skindata/PAD/images'),
        ]
        for name,csv_path,root in checks:
            miss=0; total=0
            with open(csv_path,newline='',encoding='utf-8') as f:
                r=csv.DictReader(f)
                for row in r:
                    total += 1
                    p=(row.get('image') or '').strip()
                    ok=os.path.isfile(p) or os.path.isfile(os.path.join(root,p))
                    if not ok:
                        miss += 1
            print(f'{name}: total={total}, missing={miss}')
        PY

Expected result on this machine:

- Patch16: total=2804, missing=0
- HAM10K: total=1512, missing=0
- PAD: total=2298, missing=0

## 5. Run Example

        cd /mnt/data/ssz/Skin/SkinMR
        DATASET_KEY="pad" bash ZS_classify_test.sh

## 6. Notes

- If a download is interrupted, delete only the broken .part file and rerun.
- Keep original filename case and extensions unchanged.
- Do not remove Dataframe/test/classification/*.csv.
