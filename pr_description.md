🎯 **What:** The `iso_num_to_alpha3` helper function was duplicated exactly across six different script files in `analysis/figures/`. It was consolidated into a new module, `analysis/shared/utils.py`, and imported where needed. The redundant `import pycountry` statements were also cleaned up.
💡 **Why:** This reduces code duplication and improves maintainability. Any future changes or bug fixes to this helper function now only need to be done in one place.
✅ **Verification:** Verified by confirming the new `analysis/shared/utils.py` contains the function, that old definitions were deleted, that tests run successfully, and by manually reviewing the git diff.
✨ **Result:** A cleaner codebase with fewer redundant lines, preserving all existing behavior.
