import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog, QScrollArea
import statsmodels.api as sm


class PlotPopupWindow(QDialog):
    def __init__(self, fig, min_width=800, min_height=600, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Details (Zoom & Scroll)")
        self.resize(1000, 800)
        layout = QVBoxLayout(self)

        # Create the canvas with the passed figure
        self.canvas = FigureCanvas(fig)

        # Enforce minimum size to trigger scrollbars if the plot gets too large
        self.canvas.setMinimumSize(min_width, min_height)

        # Attach standard Matplotlib zooming/panning tools
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Place the canvas inside a scrollable area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll)


class AnalysisPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None
        self.current_fig = None

        # State management for the pop-out functionality
        self.last_func = None
        self.last_args = tuple()

        sns.set_style("whitegrid")

    def _set_figure(self, fig):
        """Clears the old canvas and sets the new figure into the UI layout."""
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.current_fig = fig
        self.canvas = FigureCanvas(fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()

    def pop_out(self):
        """Generates a fresh, un-squished version of the current plot in a scrollable window."""
        if self.last_func is None:
            return

        # Regenerate figure for the popout so it isn't linked to the squished main UI canvas
        fig = self.last_func(*self.last_args)

        # Determine dynamic scroll bounds based on plot type and 'k' factors
        min_w, min_h = 800, 600
        if self.last_func.__name__ == '_generate_all_residuals':
            k = len(self.last_args[1])  # feature_names is at index 1
            min_h = 350 * (k + 1)
        elif self.last_func.__name__ == '_generate_pairplot':
            k = len(self.last_args[2])  # feature_names is at index 2
            min_w = 300 * (k + 1)
            min_h = 300 * (k + 1)
        elif self.last_func.__name__ == '_generate_subset_chart':
            min_w, min_h = 1000, 700

        self.popup = PlotPopupWindow(fig, min_width=min_w, min_height=min_h)
        self.popup.show()

    # --- Plot Generators ---
    def _generate_actual_vs_predicted(self, results, target_name):
        fig, ax = plt.subplots(tight_layout=True)
        y_actual = results.model.endog
        y_predicted = results.fittedvalues
        sns.scatterplot(x=y_actual, y=y_predicted, ax=ax, alpha=0.6, color='#2c3e50')
        mn, mx = y_actual.min(), y_actual.max()
        ax.plot([mn, mx], [mn, mx], color='#e74c3c', linestyle='--', lw=2, label='Perfect Prediction')
        ax.set_xlabel(f"Actual {target_name}", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"Predicted {target_name}", fontsize=10, fontweight='bold')
        ax.set_title("Actual vs. Predicted Values", fontsize=12)
        ax.legend()
        return fig

    def _generate_all_residuals(self, results, feature_names):
        k = len(feature_names)
        # Default sizing scales vertically based on k
        fig, axes = plt.subplots(nrows=k + 1, ncols=1, figsize=(6, 3.5 * (k + 1)), tight_layout=True)
        if k == 0:
            axes = [axes]

        fitted_vals = results.fittedvalues
        residuals = results.resid

        # 1. Top Plot: Overall Residuals vs Fitted
        sns.scatterplot(x=fitted_vals, y=residuals, ax=axes[0], alpha=0.6, color='#3498db')
        axes[0].axhline(0, color='#e74c3c', linestyle='--', lw=2)
        axes[0].set_xlabel("Fitted Values", fontsize=10, fontweight='bold')
        axes[0].set_ylabel("Residuals", fontsize=10, fontweight='bold')
        axes[0].set_title("Residuals vs. Fitted", fontsize=12)

        # 2. Subsequent Plots: Residuals vs each Independent Variable
        exog_df = results.model.data.orig_exog
        for i, feature in enumerate(feature_names):
            ax = axes[i + 1]
            x_vals = exog_df[feature]
            sns.scatterplot(x=x_vals, y=residuals, ax=ax, alpha=0.6, color='#9b59b6')
            ax.axhline(0, color='#e74c3c', linestyle='--', lw=2)
            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel("Residuals", fontsize=10, fontweight='bold')
            ax.set_title(f"Residuals vs. {feature}", fontsize=12)

        return fig

    def _generate_pairplot(self, df, target_name, feature_names):
        cols = feature_names + [target_name]
        plot_df = df[cols]
        # corner=True creates the clean lower-triangle style
        g = sns.pairplot(plot_df, corner=True, kind='reg',
                         plot_kws={'line_kws': {'color': '#e74c3c'}, 'scatter_kws': {'alpha': 0.6, 'color': '#16a085'}},
                         diag_kws={'color': '#16a085'})
        fig = g.fig
        fig.subplots_adjust(top=0.95)
        fig.suptitle(f"Pairplot of Factors vs {target_name}", fontsize=14)
        return fig

    def _generate_qq(self, results):
        fig, ax = plt.subplots(tight_layout=True)
        sm.qqplot(results.resid, line='s', ax=ax, markerfacecolor='#3498db', markeredgecolor='#2980b9', alpha=0.6)
        ax.set_title("Normal Q-Q Plot of Residuals", fontsize=12)
        return fig

    # --- Public Triggers ---
    def plot_actual_vs_predicted(self, results, target_name: str):
        self.last_func = self._generate_actual_vs_predicted
        self.last_args = (results, target_name)
        self._set_figure(self.last_func(*self.last_args))

    def plot_all_residuals(self, results, feature_names: list):
        self.last_func = self._generate_all_residuals
        self.last_args = (results, feature_names)
        self._set_figure(self.last_func(*self.last_args))

    def plot_pairplot(self, df: pd.DataFrame, target_name: str, feature_names: list):
        self.last_func = self._generate_pairplot
        self.last_args = (df, target_name, feature_names)
        self._set_figure(self.last_func(*self.last_args))

    def plot_qq(self, results):
        self.last_func = self._generate_qq
        self.last_args = (results,)
        self._set_figure(self.last_func(*self.last_args))

    def _generate_subset_chart(self, metrics_df, total_features):
        # Sort by BIC (lowest to highest) to find the "elbow"
        df_sorted = metrics_df.sort_values(by='BIC').reset_index(drop=True)

        fig, ax1 = plt.subplots(figsize=(10, 6), tight_layout=True)

        # Left Y-axis (AIC, BIC) - Standard Quant Colors
        color_aic = '#00BFFF'
        color_bic = '#0047AB'

        ax1.set_xlabel('Models sorted by lowest to highest BIC', fontweight='bold')
        ax1.set_ylabel('AIC, BIC', fontweight='bold')
        line1, = ax1.plot(df_sorted['Model'], df_sorted['AIC'], color=color_aic, lw=2.5, label='AIC')
        line2, = ax1.plot(df_sorted['Model'], df_sorted['BIC'], color=color_bic, lw=2.5, label='BIC')

        # Rotate X labels 90 degrees so the string labels don't overlap
        ax1.tick_params(axis='x', rotation=90, labelsize=8)

        # Right Y-axis (R2, Adj R2)
        ax2 = ax1.twinx()
        color_r2 = '#00FA9A'
        color_adj = '#DC143C'

        ax2.set_ylabel('R2 or adjusted R2', fontweight='bold')
        line3, = ax2.plot(df_sorted['Model'], df_sorted['R2'], color=color_r2, lw=2.5, label='R2')
        line4, = ax2.plot(df_sorted['Model'], df_sorted['Adj_R2'], color=color_adj, lw=2.5, label='Adjusted R2')

        # Consolidate legends from both axes
        lines = [line1, line2, line3, line4]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)

        total_models = len(metrics_df)
        fig.suptitle(
            f"All {total_models} Models of Excess Portfolio Returns\nRegressed on Up to {total_features} Factors",
            fontsize=14)

        return fig

    # Add the public trigger for the new chart
    def plot_subset_chart(self, metrics_df: pd.DataFrame, total_features: int):
        self.last_func = self._generate_subset_chart
        self.last_args = (metrics_df, total_features)
        self._set_figure(self.last_func(*self.last_args))