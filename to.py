# Usage: to.py notebook_name.ipynb new_python_file_name.py
import re
import argparse
import nbconvert


parser = argparse.ArgumentParser()

# first positional argument is the path to the notebook
parser.add_argument('notebook', type=str, help='path to the notebook', default='main-train.ipynb')

# second positional argument is the path to the output file
parser.add_argument('output', type=str, help='path to the output file', default='main-train.py')

# args = ['notebook_name.ipynb', 'new_python_file_name.py']  
args = parser.parse_args()

# convert notebook to python script string
file = nbconvert.PythonExporter().from_filename(args.notebook)[0]

# replace '# In[ ]:\n\n' with '# %%'
file = re.sub('# In\[([\d]+| )\]:\n\n', '# %%', file)

# removed the three lines at the top of the file given by
#!/usr/bin/env python
# coding: utf-8
file = file.replace('#!/usr/bin/env python\n# coding: utf-8\n\n', '')

with open(args.output, 'w') as f:
    f.write(file)