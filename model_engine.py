import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import itertools


class RegressionEngine:
    def __init__(self):
        pass

    def run_ols(self, df: pd.DataFrame, target_col: str, feature_cols: list):
        y = df[target_col]
        X = df[feature_cols]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        return results

    def get_summary_text(self, results) -> str:
        return results.summary().as_text()

    def get_anova_table(self, results) -> str:
        """Extracts degrees of freedom, sum of squares, and mean squares into an ANOVA table."""
        df_mod, df_res = int(results.df_model), int(results.df_resid)
        df_tot = int(results.nobs - 1)

        ss_mod, ss_res = results.ess, results.ssr
        ss_tot = results.centered_tss

        ms_mod, ms_res = results.mse_model, results.mse_resid
        f_stat, p_val = results.fvalue, results.f_pvalue

        table = "ANOVA Table\n"
        table += "=" * 85 + "\n"
        table += f"{'Source':<15} | {'df':<5} | {'SS':<15} | {'MS':<15} | {'F':<10} | {'Significance F':<15}\n"
        table += "-" * 85 + "\n"
        table += f"{'Regression':<15} | {df_mod:<5} | {ss_mod:<15.4f} | {ms_mod:<15.4f} | {f_stat:<10.4f} | {p_val:<15.4e}\n"
        table += f"{'Residual':<15} | {df_res:<5} | {ss_res:<15.4f} | {ms_res:<15.4f} | {'':<10} | {'':<15}\n"
        table += f"{'Total':<15} | {df_tot:<5} | {ss_tot:<15.4f} | {'':<15} | {'':<10} | {'':<15}\n"
        table += "=" * 85 + "\n"

        return table

    def run_all_subsets(self, df: pd.DataFrame, target_col: str, feature_cols: list):
        """Generates 2^k - 1 regression models and returns their metrics and ANOVA tables."""
        y = df[target_col]

        # Map feature names to indices (1 to k) to keep chart labels clean
        feature_mapping = {feat: str(i + 1) for i, feat in enumerate(feature_cols)}

        metrics_list = []
        subset_anovas = ""

        # Loop through every possible combination size (1 to k)
        for k_subset in range(1, len(feature_cols) + 1):
            for combo in itertools.combinations(feature_cols, k_subset):
                X_subset = sm.add_constant(df[list(combo)])
                model = sm.OLS(y, X_subset)
                res = model.fit()

                label = " ".join([feature_mapping[f] for f in combo])

                metrics_list.append({
                    'Model': label,
                    'AIC': res.aic,
                    'BIC': res.bic,
                    'R2': res.rsquared,
                    'Adj_R2': res.rsquared_adj
                })

                subset_anovas += f"Variables: {', '.join(combo)}\n"
                subset_anovas += f"Label: {label}\n"
                subset_anovas += self.get_anova_table(res)
                subset_anovas += "\n\n"

        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df, subset_anovas, feature_mapping

    def run_diagnostics(self, results) -> dict:
        diagnostics = {}
        bp_test = het_breuschpagan(results.resid, results.model.exog)
        lm_p_value = bp_test[1]

        diagnostics['Breusch-Pagan'] = {
            'Test': 'Breusch-Pagan',
            'Hypothesis Check': 'Homoscedasticity (Constant Variance)',
            'LM Statistic': round(bp_test[0], 4),
            'LM p-value': round(lm_p_value, 4),
            'Conclusion': 'Warning: Heteroscedasticity Detected' if lm_p_value < 0.05 else 'Pass: Homoscedasticity Assumed'
        }
        return diagnostics