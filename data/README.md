# Multilingual Search Engine - Data Directory

Dataset files (`*.csv`, `ta_rom`, etc.) are excluded from version control to maintain a lightweight repository and comply with size limits.

## Dataset Setup

### Automated Generation via Hugging Face
To download and split the Hindi and Telugu datasets from Hugging Face:
```bash
python scripts/process_hf_datasets.py
```
This produces:
- `data/hindi.csv` (Native Devanagari script)
- `data/hindi_romanized.csv` (Romanized transliteration)
- `data/telugu.csv` (Native Telugu script)
- `data/telugu_romanized.csv` (Romanized transliteration)

### Supported Languages
Place language frequency CSV files in this directory with columns `word,count`:
- `english.csv`
- `tamil.csv` / `tamil_small.csv` / `tamil_romanized.csv`
- `hindi.csv` / `hindi_romanized.csv`
- `telugu.csv` / `telugu_romanized.csv`
- European languages: `spanish.csv`, `french.csv`, `german.csv`, `italian.csv`, `dutch.csv`, `portuguese.csv`
