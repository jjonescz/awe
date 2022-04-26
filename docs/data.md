# Dataset preparation

The data loading code assumes that the SWDE dataset is in folder `data/swde`
and the Apify dataset is in folder `data/apify`.
Only the former is currently open-source.
Upon first use (e.g., as part of [training](train.md)),
the datasets will be preprocessed into SQLite files.

## SWDE dataset

To get the original SWDE dataset, follow these steps.

1. Download `swde.zip` from Internet Archive.

   ```bash
   wget https://web.archive.org/web/20210630013015id_/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip
   ```

2. Extract HTML files into `data/swde/data/<vertical>/<website>/<page>.htm`.

   ```bash
   mkdir -p data/swde/data
   unzip swde.zip -d data/swde
   rm swde.zip
   unzip data/swde/sourceCode/sourceCode.zip -d data/swde/data
   rm data/swde/sourceCode/sourceCode.zip
   for f in data/swde/data/*.7z
   do
     d=${f%.7z}
     7zz x $f -o $d
     rm $f
   done
   ```

3. Extract visuals using the [visual extractor](extractor.md).

Alternatively, download pre-extracted visuals along with HTML files:

```bash
git clone https://github.com/jjonescz/swde-visual data/swde
```

## GloVe

Pre-trained GloVe embeddings are downloaded automatically before first use.
To pre-download them, execute:

```bash
python -m awe.data.glove
```
