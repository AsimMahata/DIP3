import json
import os

input_file = r"c:\Users\asim\projects\project3Dip\pipeline\multilingual_abuse_detection_11_04_all_lang.ipynb"
output_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace NUM_LANGUAGES
            if 'NUM_LANGUAGES       = 3' in line:
                line = line.replace('3', '12')
            
            # Replace LANG_TO_IDX
            if 'LANG_TO_IDX         = {"english": 0, "hinglish": 1, "banglish": 2}' in line:
                line = line.replace('{"english": 0, "hinglish": 1, "banglish": 2}',
                                    '{"english": 0, "hinglish": 1, "banglish": 2, "marathi": 3, "hindi": 4, "kanglish": 5, "kannada": 6, "tamil": 7, "tanglish": 8, "manglish": 9, "malayalam": 10, "bangla": 11}')
            
            # Replace length checks
            if 'if len(raw_dfs) != 3:' in line:
                line = line.replace('3', '12')
            if 'Expected 3 datasets' in line:
                line = line.replace('3 datasets', '12 datasets')
            if 'Target total: {3 *' in line:
                line = line.replace('{3 *', '{12 *')
            if '(3 ×' in line:
                line = line.replace('(3 ×', '(12 ×')
            
            # Update languages for abbrevs
            if 'if lang in ("hinglish", "banglish"):' in line:
                line = line.replace('("hinglish", "banglish")', 
                                    '("hinglish", "banglish", "kanglish", "tanglish", "manglish", "marathi", "hindi", "kannada", "tamil", "malayalam", "bangla")')
            
            new_source.append(line)
        cell['source'] = new_source

# Now rewrite the DESI_ABBREVS definition in the source.
# The user wants to keep only true abbreviations like bc, mc, bsdk, bhenchod.
desi_abbrevs_lines = [
    'DESI_ABBREVS = {\n',
    '    "bc": "behanchod",\n',
    '    "mc": "madarchod",\n',
    '    "bhenchod": "sisterfucker",\n',
    '    # Added generic desi text abbreviations\n',
    '    "mkc": "maa ki chut",\n',
    '    "tmkc": "teri maa ki chut",\n',
    '    "bck": "behenchod",\n',
    '    "pk": "pagal",\n',
    '    "gnd": "gandu",\n',
    '    "chtiya": "chutiya",\n',
    '    "bchd": "behenchod",\n',
    '}\n'
]

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('DESI_ABBREVS = {' in line for line in source):
            # Find start and end of DESI_ABBREVS
            start_idx = None
            end_idx = None
            for i, line in enumerate(source):
                if line.startswith('DESI_ABBREVS = {'):
                    start_idx = i
                if start_idx is not None and i > start_idx and line.startswith('}'):
                    end_idx = i
                    break
            
            if start_idx is not None and end_idx is not None:
                new_source = source[:start_idx] + desi_abbrevs_lines + source[end_idx+1:]
                cell['source'] = new_source

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Modified notebook saved as final_notebook.ipynb")
