import numpy as np
import pandas as pd

from hansard_pm_nlp.regression import (
    add_bh_correction,
    extract_crisis_effects,
    extract_interaction_effect,
    fit_h2_model,
    fit_h3_model,
)


def _toy_docs():
    rng = np.random.default_rng(42)
    rows = []
    # Johnson: baseline + covid crisis (clearly more negative sentiment)
    for i in range(20):
        rows.append(
            {
                "pm_name": "Boris Johnson",
                "pm_party": "Conservative",
                "any_crisis": False,
                "crisis_covid19": False,
                "crisis_mini_budget": False,
                "crisis_ukraine_invasion": False,
                "crisis_labour_leadership_crisis": False,
                "sentiment": rng.normal(0.2, 0.05),
            }
        )
    for i in range(20):
        rows.append(
            {
                "pm_name": "Boris Johnson",
                "pm_party": "Conservative",
                "any_crisis": True,
                "crisis_covid19": True,
                "crisis_mini_budget": False,
                "crisis_ukraine_invasion": False,
                "crisis_labour_leadership_crisis": False,
                "sentiment": rng.normal(-0.3, 0.05),
            }
        )
    # Starmer: baseline + labour crisis (no sentiment shift, to test interaction)
    for i in range(20):
        rows.append(
            {
                "pm_name": "Keir Starmer",
                "pm_party": "Labour",
                "any_crisis": False,
                "crisis_covid19": False,
                "crisis_mini_budget": False,
                "crisis_ukraine_invasion": False,
                "crisis_labour_leadership_crisis": False,
                "sentiment": rng.normal(0.2, 0.05),
            }
        )
    for i in range(20):
        rows.append(
            {
                "pm_name": "Keir Starmer",
                "pm_party": "Labour",
                "any_crisis": True,
                "crisis_covid19": False,
                "crisis_mini_budget": False,
                "crisis_ukraine_invasion": False,
                "crisis_labour_leadership_crisis": True,
                "sentiment": rng.normal(0.2, 0.05),
            }
        )
    return pd.DataFrame(rows)


def test_fit_h2_model_detects_covid_effect():
    docs = _toy_docs()
    result = fit_h2_model(docs, "sentiment")
    effects = extract_crisis_effects(result, "sentiment")
    covid_row = effects[effects["crisis"] == "covid19"].iloc[0]
    assert covid_row["coef"] < -0.3
    assert covid_row["pvalue"] < 0.05


def test_fit_h3_model_detects_differential_party_effect():
    docs = _toy_docs()
    result = fit_h3_model(docs, "sentiment")
    interaction = extract_interaction_effect(result, "sentiment")
    # Johnson's crisis effect is strongly negative, Starmer's is ~zero, so
    # the Labour-differential term should be positive and significant.
    assert interaction["coef"] > 0.3
    assert interaction["pvalue"] < 0.05


def test_add_bh_correction_adds_expected_columns():
    effects = pd.DataFrame({"pvalue": [0.001, 0.04, 0.5, 0.9]})
    corrected = add_bh_correction(effects)
    assert "pvalue_bh" in corrected.columns
    assert "significant_bh" in corrected.columns
    assert corrected["pvalue_bh"].iloc[0] <= corrected["pvalue_bh"].iloc[-1]


def test_add_bh_correction_never_less_significant_than_raw():
    effects = pd.DataFrame({"pvalue": [0.01, 0.02, 0.03]})
    corrected = add_bh_correction(effects)
    assert (corrected["pvalue_bh"] >= corrected["pvalue"]).all()
