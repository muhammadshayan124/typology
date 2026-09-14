"""Per-transaction explanation for flagged transactions -- the real regulatory need this
addresses: an AML alert that can't be explained to a compliance officer (or eventually a
regulator) isn't actionable, no matter how accurate the underlying model is. Uses real
SHAP values (TreeExplainer, exact for tree ensembles like XGBoost), not a proxy like
generic feature importance -- SHAP explains *this specific prediction*, feature importance
only explains the model in aggregate.
"""

import pandas as pd
import shap
from xgboost import XGBClassifier

from typology.model import FEATURE_COLUMNS


def explain_flagged(model: XGBClassifier, features: pd.DataFrame, tx_id: int, top_n: int = 5) -> dict:
    """Returns the top contributing features (by |SHAP value|) for why this specific
    transaction was scored the way it was -- a human-readable audit trail per flag.
    """
    row = features.loc[[tx_id], FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    contributions = list(zip(FEATURE_COLUMNS, shap_values[0], row.iloc[0].tolist()))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    return {
        "tx_id": tx_id,
        "score": float(model.predict_proba(row)[0, 1]),
        "top_factors": [
            {"feature": name, "value": float(value), "shap_contribution": float(shap_val)}
            for name, shap_val, value in contributions[:top_n]
        ],
    }
