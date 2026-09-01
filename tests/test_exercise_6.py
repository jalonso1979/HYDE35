import pandas as pd
from unittest.mock import patch, MagicMock
from analysis.paper4_shadow.run_all import exercise_6

def test_exercise_6_figures_only():
    climate_df = pd.DataFrame({
        "region": [1, 2],
        "year": [1800, 1800],
        "temperature_c": [10, 20],
        "precipitation_mm": [2, 3]
    })

    data = {
        "climate_panel": climate_df
    }
    with patch("analysis.paper4_shadow.run_all._try_figure") as mock_fig:
        exercise_6(data, figures_only=True)
