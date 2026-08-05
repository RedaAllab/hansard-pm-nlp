"""Event-study regressions (Phase 7): tests H2 (crisis affect) and H3 (party
interaction) via OLS with PM fixed effects and HC3 robust standard errors.

H2: does each named crisis shift a PM's affect score relative to their own
baseline? DV ~ PM fixed effects + one dummy per crisis window.

H3: does the shift during a crisis differ by governing party? Per-crisis x
party interactions aren't identifiable (event_study.py: each window overlaps
exactly one party), so H3 uses a pooled any_crisis dummy, parametrized so a
single coefficient - is_labour:any_crisis - is the interaction effect
directly: DV ~ PM fixed effects + any_crisis (the Conservative crisis effect,
Conservative being the reference party) + is_labour:any_crisis (the
additional/differential effect for Labour).
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.multitest import multipletests

from hansard_pm_nlp.event_study import CRISIS_WINDOWS

CRISIS_COLUMNS = [f"crisis_{name}" for name in CRISIS_WINDOWS]


def fit_h2_model(docs: pd.DataFrame, dv: str) -> RegressionResultsWrapper:
    docs = docs.copy()
    docs[CRISIS_COLUMNS] = docs[CRISIS_COLUMNS].astype(int)
    formula = f"{dv} ~ C(Q('pm_name')) + " + " + ".join(CRISIS_COLUMNS)
    return smf.ols(formula, data=docs).fit(cov_type="HC3")


def fit_h3_model(docs: pd.DataFrame, dv: str) -> RegressionResultsWrapper:
    docs = docs.copy()
    docs["is_labour"] = (docs["pm_party"] == "Labour").astype(int)
    docs["any_crisis"] = docs["any_crisis"].astype(int)
    formula = f"{dv} ~ C(Q('pm_name')) + any_crisis + is_labour:any_crisis"
    return smf.ols(formula, data=docs).fit(cov_type="HC3")


def extract_crisis_effects(result: RegressionResultsWrapper, dv: str) -> pd.DataFrame:
    rows = []
    for col in CRISIS_COLUMNS:
        crisis_name = col.removeprefix("crisis_")
        rows.append(
            {
                "dv": dv,
                "crisis": crisis_name,
                "coef": result.params[col],
                "se": result.bse[col],
                "pvalue": result.pvalues[col],
            }
        )
    return pd.DataFrame(rows)


def extract_interaction_effect(result: RegressionResultsWrapper, dv: str) -> dict:
    term = "is_labour:any_crisis"
    return {
        "dv": dv,
        "term": term,
        "coef": result.params[term],
        "se": result.bse[term],
        "pvalue": result.pvalues[term],
    }


def add_bh_correction(effects: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction across all rows in `effects` (one
    hypothesis family per call - H2's 12 tests and H3's 3 tests are
    corrected separately, not pooled together).
    """
    effects = effects.copy()
    reject, adj_pvalues, _, _ = multipletests(effects["pvalue"], alpha=alpha, method="fdr_bh")
    effects["pvalue_bh"] = adj_pvalues
    effects["significant_bh"] = reject
    return effects
