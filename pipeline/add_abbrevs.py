import json
import os

input_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"
output_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"

# Full expanded dictionary string format
expanded_abbrevs_lines = [
    'DESI_ABBREVS = {\n',
    '    # --- Generic Hinglish & North Indian ---\n',
    '    "bc": "behanchod",\n',
    '    "mc": "madarchod",\n',
    '    "bhenchod": "sisterfucker",\n',
    '    "mkc": "maa ki chut",\n',
    '    "tmkc": "teri maa ki chut",\n',
    '    "bck": "behenchod",\n',
    '    "pk": "pagal",\n',
    '    "gnd": "gandu",\n',
    '    "chtiya": "chutiya",\n',
    '    "bchd": "behenchod",\n',
    '    "bsdk": "bhosdike",\n',
    '    \n',
    '    # --- Banglish (Bengali) Slangs ---\n',
    '    "bcoda": "bokachoda",\n',
    '    "sb": "suorer baccha",\n',
    '    "kc": "khankir chele",\n',
    '    "mgi": "magi",\n',
    '    "bal": "baal",     # often meaning bullshit/nonsense\n',
    '    \n',
    '    # --- Tanglish (Tamil) Slangs ---\n',
    '    "otha": "ommala",\n',
    '    "ommale": "ommaala",\n',
    '    "punda": "pundai",\n',
    '    "tp": "thevidiya paiyan",\n',
    '    "mairu": "myre",\n',
    '    "gotha": "gothaa",\n',
    '    \n',
    '    # --- Manglish (Malayalam) Slangs ---\n',
    '    "myr": "myre",\n',
    '    "thendi": "beggar",\n',
    '    "thayoli": "motherfucker",\n',
    '    "pooran": "asshole",\n',
    '    \n',
    '    # --- Kanglish (Kannada) Slangs ---\n',
    '    "bm": "boli magane",\n',
    '    "sm": "sule magane",\n',
    '    "shata": "shata",\n',
    '    "loosu": "loosu",\n',
    '    \n',
    '    # --- Marathi Slangs ---\n',
    '    "aiz": "aizavadya",\n',
    '    "lavdya": "lavdya",\n',
    '    "bhadkya": "bhadkhau",\n',
    '    "raand": "raand",\n',
    '}\n'
]

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        start_idx = None
        end_idx = None
        for i, line in enumerate(source):
            if line.startswith('DESI_ABBREVS = {'):
                start_idx = i
            if start_idx is not None and i > start_idx and line.startswith('}'):
                end_idx = i
                break
        
        if start_idx is not None and end_idx is not None:
            new_source = source[:start_idx] + expanded_abbrevs_lines + source[end_idx+1:]
            cell['source'] = new_source

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Expanded DESI_ABBREVS successfully integrated into the notebook.")
