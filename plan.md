1. **Understand & Assess**: Refactor `exercise_6` in `analysis/paper4_shadow/run_all.py` to extract its logic into smaller, well-named helper functions (`_prepare_seasonality_endowments`, `_build_and_run_cross_section`, `_plot_long_shadow_figures`).
2. **Implement**:
   - Define `_prepare_seasonality_endowments` to build seasonality panels and rename columns.
   - Define `_build_and_run_cross_section` to build cross section and run regressions.
   - Define `_plot_long_shadow_figures` to handle the figure plotting.
   - Update `exercise_6` to use these helper functions while preserving logic and condition (`figures_only`).
   - Done.
3. **Verify**: Run `python -m py_compile analysis/paper4_shadow/run_all.py` to check for syntax errors. Then run `python -c "import patch_config; import pytest; pytest.main(['tests/test_exercise_6.py'])"` to ensure test passes. Checked flake8 output in the refactored region and everything matches style guidelines (E302, E501, W293 sorted out).
4. **Pre-commit and PR**: Call pre-commit instructions, run whatever is needed, submit.
