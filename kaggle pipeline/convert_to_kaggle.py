import json

notebook_path = r"c:\Users\asim\projects\project3Dip\kaggle pipeline\final_notebook_nlp_dip_sem6.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = cell['source']
    if not source:
        continue
    
    # Check Section 2 (Config)
    if any('SECTION 2 — CONFIG' in line for line in source):
        # We replace DRIVE_ROOT
        for j, line in enumerate(source):
            if line.startswith('DRIVE_ROOT          = '):
                source[j] = 'DRIVE_ROOT          = "/kaggle/working"\n'
                
    # Check Section 3 (Drive Mount)
    if any('SECTION 3 — DRIVE MOUNT + ORGANISED FOLDER SETUP' in line for line in source):
        # We remove drive import and mount
        for j in range(len(source)):
            if 'google.colab' in source[j]:
                source[j] = '# ' + source[j]
            if 'drive.mount' in source[j]:
                source[j] = '# ' + source[j]
                
    # Check File Upload (Cell 5 equivalent)
    if any('from google.colab import files as colab_files' in line for line in source):
        # We rewrite this cell completely
        new_source = [
            "import os\n",
            "import glob\n",
            "import pandas as pd\n",
            "from tqdm import tqdm\n",
            "\n",
            "SEED = 42\n",
            "\n",
            "# In Kaggle, datasets are usually attached to the notebook in /kaggle/input/\n",
            "# To use this, add your .tsv or .csv datasets via the Kaggle UI (Add Data > Upload)\n",
            "dataset_paths = glob.glob(\"/kaggle/input/**/*.tsv\", recursive=True) + \\\n",
            "                glob.glob(\"/kaggle/input/**/*.csv\", recursive=True)\n",
            "\n",
            "raw_dfs = {}\n",
            'REQUIRED_COLS = {"text", "offensive", "language"}\n',
            "\n",
            "print(f\"Found {len(dataset_paths)} dataset files in /kaggle/input/\")\n",
            "\n",
            "for fname in dataset_paths:\n",
            "    try:\n",
            "        df = pd.read_csv(fname, sep=\"\\t\")\n",
            "        df.columns = df.columns.str.strip().str.lower()\n",
            "\n",
            "        missing = REQUIRED_COLS - set(df.columns)\n",
            "        if missing:\n",
            "            print(f\"{fname} missing columns: {missing} — skipped\")\n",
            "            continue\n",
            "\n",
            "        df[\"text\"] = df[\"text\"].astype(str).str.strip()\n",
            "        df[\"offensive\"] = df[\"offensive\"].astype(int)\n",
            "        df[\"language\"] = df[\"language\"].astype(str).str.strip().str.lower()\n",
            "\n",
            "        print(f\"\\n{fname} loaded:\")\n",
            "        print(df[\"language\"].value_counts())\n",
            "\n",
            "        for lang in df[\"language\"].unique():\n",
            "            part = df[df[\"language\"] == lang].copy()\n",
            "            raw_dfs.setdefault(lang, []).append(part)\n",
            "\n",
            "    except Exception as e:\n",
            "        print(f\"{fname} failed to parse: {e}\")\n",
            "\n",
            "# merge same-language parts\n",
            "raw_dfs = {\n",
            "    lang: pd.concat(parts, ignore_index=True)\n",
            "    for lang, parts in raw_dfs.items()\n",
            "}\n",
            "\n",
            "# mappings\n",
            "unique_langs = sorted(list(raw_dfs.keys()))\n",
            "NUM_LANGUAGES = max(len(unique_langs), 2)\n",
            "LANG_TO_IDX = {lang: idx for idx, lang in enumerate(unique_langs)}\n",
            "IDX_TO_LANG = {v: k for k, v in LANG_TO_IDX.items()}\n",
            "\n",
            "print(f\"\\nDetected {len(unique_langs)} languages: {unique_langs}\\n\")\n",
            "\n",
            "if len(unique_langs) == 0:\n",
            "    raise ValueError(\"No valid datasets loaded.\")\n",
            "\n",
            "df_all = pd.concat(raw_dfs.values(), ignore_index=True)\n",
            "df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)\n",
            "\n",
            "df_all[\"lang_id\"] = df_all[\"language\"].map(LANG_TO_IDX)\n",
            "\n",
            "print(f\"\\nMerged dataset: {len(df_all):,} rows\")\n",
            "print(f\"Overall offensive rate: {df_all['offensive'].mean()*100:.1f}%\")\n",
            "print(\"\\nLanguage distribution:\")\n",
            "print(df_all[\"language\"].value_counts().to_string())\n",
            "\n",
            "df_all.to_csv(\"final_dataset.tsv\", sep=\"\\t\", index=False)\n",
            "print(\"\\nSaved as final_dataset.tsv\")\n"
        ]
        cell['source'] = new_source

# Create new zip block cell at the end of the notebook!
zip_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "\n",
    "# SECTION 17 — ZIP OUTPUTS FOR EASY DOWNLOAD\n",
    "\n",
    "import shutil\n",
    "\n",
    "print(\"Zipping /kaggle/working directory...\")\n",
    "shutil.make_archive(\"/kaggle/working/multilingual_abuse_detection_outputs\", \"zip\", \"/kaggle/working/\")\n",
    "print(\"\\nAll outputs zipped to /kaggle/working/multilingual_abuse_detection_outputs.zip for easy download!\")\n"
  ]
}

nb['cells'].append(zip_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook converted successfully!")
