"""Data-quality checks run by the worker (Data-Audit A7 label issues, A8 fairness).

Needs the `quality` extra (Cleanlab, Fairlearn, scikit-learn); label-issue detection uses LightGBM when the
`tabular` extra is installed too. The CLI imports these modules lazily.
"""
