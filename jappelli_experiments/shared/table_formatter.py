"""
LaTeX / HTML / CSV table output for regression results.

Provides consistent formatting across all experiments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from jappelli_experiments.config import TABLE_DIR


def _stars(p):
    """Return significance stars."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def format_regression_table(results_list, model_names=None, dep_var="",
                            note="", digits=3):
    """
    Format multiple regression results into a publication-style table.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys: coef (Series), se (Series), p_value (Series),
        n_obs (int), r2 (float).
    model_names : list of str
        Column headers for each model.
    dep_var : str
        Dependent variable name.
    note : str
        Table footnote.
    digits : int
        Decimal places.

    Returns
    -------
    DataFrame formatted for display.
    """
    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(results_list))]

    # Collect all variable names
    all_vars = []
    for r in results_list:
        for v in r["coef"].index:
            if v not in all_vars:
                all_vars.append(v)

    rows = []
    for var in all_vars:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}

        for name, r in zip(model_names, results_list):
            if var in r["coef"].index:
                c = r["coef"][var]
                s = r["se"][var]
                p = r["p_value"][var]
                coef_row[name] = f"{c:.{digits}f}{_stars(p)}"
                se_row[name] = f"({s:.{digits}f})"
            else:
                coef_row[name] = ""
                se_row[name] = ""

        rows.append(coef_row)
        rows.append(se_row)

    # Add N and R2
    n_row = {"Variable": "N"}
    r2_row = {"Variable": "R²"}
    for name, r in zip(model_names, results_list):
        n_row[name] = f"{r['n_obs']:,}"
        r2_row[name] = f"{r['r2']:.{digits}f}"

    rows.append(n_row)
    rows.append(r2_row)

    return pd.DataFrame(rows)


def to_latex(df, filename, caption="", label="", note=""):
    """
    Export a formatted table to LaTeX.

    Parameters
    ----------
    df : DataFrame
        Table to export.
    filename : str
        Output filename (without path).
    caption : str
        Table caption.
    label : str
        LaTeX label.
    note : str
        Table note.
    """
    path = TABLE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    latex = df.to_latex(index=False, escape=False)

    if caption or label:
        header = "\\begin{table}[htbp]\n\\centering\n"
        if caption:
            header += f"\\caption{{{caption}}}\n"
        if label:
            header += f"\\label{{{label}}}\n"
        footer = ""
        if note:
            footer += f"\n\\vspace{{0.2cm}}\n\\footnotesize{{Note: {note}}}\n"
        footer += "\\end{table}\n"
        latex = header + latex + footer

    with open(path, "w") as f:
        f.write(latex)

    return path


def to_csv(df, filename):
    """Export table to CSV."""
    path = TABLE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def to_html(df, filename, title=""):
    """Export table to HTML."""
    path = TABLE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    html = df.to_html(index=False, escape=False)
    if title:
        html = f"<h3>{title}</h3>\n{html}"

    with open(path, "w") as f:
        f.write(html)
    return path
