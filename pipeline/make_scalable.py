import json
import os
import re

input_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"
output_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    
    source = cell['source']
    new_source = []
    
    for i, line in enumerate(source):
        # 1. Update Config (Cell 2)
        if line.startswith('NUM_LANGUAGES       ='):
            line = 'NUM_LANGUAGES       = None   # Will be set dynamically by uploaded data\n'
        elif line.startswith('LANG_TO_IDX         ='):
            line = 'LANG_TO_IDX         = {}     # Will be set dynamically\n'
        elif line.startswith('IDX_TO_LANG         ='):
            line = 'IDX_TO_LANG         = {}     # Will be set dynamically\n'
            
        # 2. Update Dataset loading (Cell 5)
        # Remove hardcoded checks
        if 'if lang_val not in LANG_TO_IDX:' in line:
            # We skip this and the next 2 lines
            continue
        if i >= 1 and 'if lang_val not in LANG_TO_IDX:' in source[i-1]:
            continue
        if i >= 2 and 'if lang_val not in LANG_TO_IDX:' in source[i-2]:
            continue
            
        # Replace ValueError checks
        if 'if len(raw_dfs) != 12:' in line:
            line = '''# dynamically create the idx mappings\nunique_langs  = sorted(list(raw_dfs.keys()))\nNUM_LANGUAGES = len(unique_langs)\nLANG_TO_IDX   = {lang: idx for idx, lang in enumerate(unique_langs)}\nIDX_TO_LANG   = {v: k for k, v in LANG_TO_IDX.items()}\n\nprint(f"\\\\nDetected {NUM_LANGUAGES} languages: {unique_langs}\\\\n")\n\nif NUM_LANGUAGES == 0:\n'''
            
        if 'raise ValueError(f"Expected 12 datasets' in line:
            line = '    raise ValueError("No valid datasets loaded.")\n'
            
        # Replace Target total log
        if 'Target total:' in line and '12 * MIN_SIZE' in line:
            line = 'print(f"   Target total: {NUM_LANGUAGES * MIN_SIZE:,} rows ({NUM_LANGUAGES} × {MIN_SIZE:,})\\n")\n'
            
        # 3. Update Preprocessing (Cell 7)
        if 'if lang in ("hinglish", "banglish"' in line:
            line = '    if lang != "english":\n'
            
        # 4. Update EDA charts (Cell 6)
        if 'color=["steelblue","salmon","gold"]' in line:
            line = line.replace('color=["steelblue","salmon","gold"]', 'color=sns.color_palette("muted", n_colors=len(lang_off))')
        if 'colors=["steelblue","salmon","gold"]' in line:
            line = line.replace('colors=["steelblue","salmon","gold"]', 'colors=sns.color_palette("muted", n_colors=len(lang_counts))')
            
        new_source.append(line)
        
    cell['source'] = new_source

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Dynamic and scalable notebook successfully saved.")
