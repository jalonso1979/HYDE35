with open("analysis/paper4_shadow/run_all.py", "r") as f:
    lines = f.readlines()

import re

for i, line in enumerate(lines):
    # Fix E302
    if i in [540, 571, 578]:
        lines[i] = "\n" + line

    # Fix W293
    if i in [568, 588]:
        lines[i] = "\n"

with open("analysis/paper4_shadow/run_all.py", "w") as f:
    f.writelines(lines)
