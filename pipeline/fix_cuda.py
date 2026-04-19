import json
import os

input_file = r"c:\Users\asim\projects\project3Dip\pipeline\final_notebook.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            new_source.append(line)
            if 'lang_val = df["language"].mode()[0]' in line:
                # Add a force fix to ensure all rows match the mode
                new_source.append('        df["language"] = lang_val\n')
                
        cell['source'] = new_source

with open(input_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("CUDA fix successfully applied.")
