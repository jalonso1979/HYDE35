with open("analysis/paper4_shadow/run_all.py", "r") as f:
    lines = f.readlines()

def wrap_line(lines, i, replace_with):
    lines[i] = replace_with

# Only care about lines we touched for the refactor.
# Our new lines start around 522 to 590.
# 522: def _prepare_seasonality_endowments(data: dict, endowments: pd.DataFrame | None) -> pd.DataFrame:
wrap_line(lines, 521, "def _prepare_seasonality_endowments(\n")
wrap_line(lines, 522, "    data: dict, endowments: pd.DataFrame | None\n")
lines.insert(523, ") -> pd.DataFrame:\n")

# 543 (was 542): def _build_and_run_cross_section(data: dict, seas_endow: pd.DataFrame) -> pd.DataFrame:
wrap_line(lines, 543, "def _build_and_run_cross_section(\n")
lines.insert(544, "    data: dict, seas_endow: pd.DataFrame\n")
lines.insert(545, ") -> pd.DataFrame:\n")

# 579: seas_col = "seasonality_historical" if "seasonality_historical" in xs.columns else "hist_seasonality_proxy"
wrap_line(lines, 579, "        seas_col = (\n")
lines.insert(580, "            \"seasonality_historical\" if \"seasonality_historical\" "
                   "in xs.columns\n")
lines.insert(581, "            else \"hist_seasonality_proxy\"\n")
lines.insert(582, "        )\n")

with open("analysis/paper4_shadow/run_all.py", "w") as f:
    f.writelines(lines)
