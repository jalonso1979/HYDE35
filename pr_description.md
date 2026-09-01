💡 **What:** Replaced the iterative `for` loop in `detect_intensification_onset` with a vectorized numpy operation utilizing a boolean mask and `argmax`. Handled edge cases including empty arrays and `NaN` values safely.

🎯 **Why:** Searching through numpy arrays with a Python loop is significantly slower than using native numpy vectorization. This optimization addresses the overhead to improve feature extraction performance.

📊 **Measured Improvement:**
In a benchmark testing a 100,000 element array where the onset condition occurs at the very end (worst-case scenario for the original code):
- **Original Time:** 16.89 seconds (for 100 executions)
- **Vectorized Time:** 0.005 seconds (for 100 executions)

This represents an enormous performance boost of over 3000x for worst-case scenarios with large arrays, while avoiding warnings natively.
