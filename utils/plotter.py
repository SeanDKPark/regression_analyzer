import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import statsmodels.api as sm


class AnalysisPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 1. Create the Matplotlib Figure and the 'canvas' (the PyQt widget holder)
        # tight_layout=True automatically handles margins for labels
        self.figure, self.ax = plt.subplots(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        # 2. Add the canvas widget to this QWidget's layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Set a clean Seaborn style for a professional look
        sns.set_style("whitegrid")

    def plot_actual_vs_predicted(self, results, target_name: str):
        """
        Draws the Actual vs. Predicted scatter plot.

        Parameters:
        - results: The fitted statsmodels results object.
        - target_name (str): The name of the Y variable for labeling.
        """
        # Clear the previous plot if one exists
        self.ax.clear()

        # 1. Extract data from the statsmodels results object
        # y_actual is found in model.endog
        y_actual = results.model.endog
        # y_predicted is found in fittedvalues
        y_predicted = results.fittedvalues

        # 2. Draw the Scatter Plot using Seaborn
        sns.scatterplot(x=y_actual, y=y_predicted, ax=self.ax, alpha=0.6, color='#2c3e50')

        # 3. Add the 45-degree reference line (Perfect Prediction Line)
        # We calculate the min/max of the actual data to define the line's start and end points
        mn = y_actual.min()
        mx = y_actual.max()
        self.ax.plot([mn, mx], [mn, mx], color='#e74c3c', linestyle='--', lw=2, label='Perfect Prediction')

        # 4. Add labels, title, and legend
        self.ax.set_xlabel(f"Actual {target_name}", fontsize=10, fontweight='bold')
        self.ax.set_ylabel(f"Predicted {target_name}", fontsize=10, fontweight='bold')
        self.ax.set_title("Actual vs. Predicted Values", fontsize=12)
        self.ax.legend()

        # 5. Crucial step: Trigger the canvas to refresh and show the new plot
        self.canvas.draw()

    def plot_residuals_vs_fitted(self, results):
        """
        Draws the Residuals vs. Fitted plot to check for homoscedasticity and non-linearity.
        """
        self.ax.clear()

        fitted_vals = results.fittedvalues
        residuals = results.resid

        # Scatter the residuals
        sns.scatterplot(x=fitted_vals, y=residuals, ax=self.ax, alpha=0.6, color='#3498db')

        # Add a horizontal line at zero
        self.ax.axhline(0, color='#e74c3c', linestyle='--', lw=2, label='Zero Error Line')

        self.ax.set_xlabel("Fitted Values", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Residuals", fontsize=10, fontweight='bold')
        self.ax.set_title("Residuals vs. Fitted", fontsize=12)
        self.ax.legend()

        self.canvas.draw()

    def plot_qq(self, results):
        """
        Draws a Normal Q-Q plot to check if the residuals are normally distributed.
        """
        self.ax.clear()

        # statsmodels has a built-in qqplot that cleanly handles the math
        # line='s' adds the standardized reference line
        sm.qqplot(results.resid, line='s', ax=self.ax,
                  markerfacecolor='#3498db', markeredgecolor='#2980b9', alpha=0.6)

        self.ax.set_title("Normal Q-Q Plot of Residuals", fontsize=12)
        # statsmodels automatically sets the X and Y labels for Q-Q plots

        self.canvas.draw()