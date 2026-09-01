with open("analysis/paper4_shadow/run_all.py", "r") as f:
    lines = f.readlines()

def wrap_line(lines, i, replace_with):
    lines[i] = replace_with

wrap_line(lines, 595, "    \"\"\"Exercise 6: Long shadow — do pre-industrial endowments \\\n")
lines.insert(596, "predict modern outcomes?\"\"\"\n")

with open("analysis/paper4_shadow/run_all.py", "w") as f:
    f.writelines(lines)
