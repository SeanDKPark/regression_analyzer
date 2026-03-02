import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


class RegressionEngine:
    def __init__(self):
        """
        Initializes the engine.
        """
        pass

    def run_ols(self, df: pd.DataFrame, target_col: str, feature_cols: list):
        """
        Runs an Ordinary Least Squares (OLS) regression.
        """
        y = df[target_col]
        X = df[feature_cols]
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        return results

    def get_summary_text(self, results) -> str:
        """
        Extracts the standard statistical summary report as a string.
        """
        return results.summary().as_text()

    def run_diagnostics(self, results) -> dict:
        """
        Runs statistical assumption tests on the fitted model.
        Returns a dictionary so the UI can modularly display specific tests.
        """
        diagnostics = {}

        # --- 1. Breusch-Pagan Test (Heteroscedasticity) ---
        # Null Hypothesis (H0): Homoscedasticity is present (variance is constant).
        # Alternative Hypothesis (HA): Heteroscedasticity is present.

        # The test requires the residuals and the original X variables (with constant)
        bp_test = het_breuschpagan(results.resid, results.model.exog)

        # bp_test returns a tuple: (LM stat, LM p-value, F-stat, F p-value)
        lm_p_value = bp_test[1]

        diagnostics['Breusch-Pagan'] = {
            'Test': 'Breusch-Pagan',
            'Hypothesis Check': 'Homoscedasticity (Constant Variance)',
            'LM Statistic': round(bp_test[0], 4),
            'LM p-value': round(lm_p_value, 4),
            # A p-value < 0.05 means we reject the null and assume heteroscedasticity
            'Conclusion': 'Warning: Heteroscedasticity Detected' if lm_p_value < 0.05 else 'Pass: Homoscedasticity Assumed'
        }

        # Future modules (like VIF for multicollinearity) will be added to this dict here.

        return diagnostics