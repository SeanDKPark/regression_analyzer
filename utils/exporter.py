import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class ReportExporter:
    def __init__(self):
        pass

    def export_to_excel(self, save_path: str, results, target_col: str, clean_df: pd.DataFrame, diagnostics_dict: dict):
        """
        Exports the regression results, diagnostics, raw data, and charts into a multi-sheet Excel file.
        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- SHEET 1: Model Summary ---
            # 1a. Core Metrics
            summary_stats = pd.DataFrame({
                'Metric': ['Dependent Variable (Y)', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
                           'No. Observations'],
                'Value': [target_col, results.rsquared, results.rsquared_adj, results.fvalue, results.f_pvalue,
                          int(results.nobs)]
            })
            summary_stats.to_excel(writer, sheet_name='Model Summary', index=False, startrow=0, startcol=0)

            # 1b. Coefficients Table
            coef_df = pd.DataFrame({
                'Coefficient': results.params,
                'Std Error': results.bse,
                't-Statistic': results.tvalues,
                'p-Value': results.pvalues
            })
            coef_df.index.name = 'Variable'
            coef_df.to_excel(writer, sheet_name='Model Summary', startrow=len(summary_stats) + 2)

            # --- SHEET 2: Diagnostics ---
            diag_rows = []
            if diagnostics_dict:
                for test_name, metrics in diagnostics_dict.items():
                    for key, value in metrics.items():
                        if key != 'Test':
                            diag_rows.append({'Assumption Test': test_name, 'Metric': key, 'Value': value})

            diag_df = pd.DataFrame(diag_rows)
            diag_df.to_excel(writer, sheet_name='Diagnostics', index=False)

            # --- SHEET 3: Model Data ---
            output_df = clean_df.copy()
            output_df['Predicted_Y'] = results.fittedvalues
            output_df['Residuals'] = results.resid
            output_df.to_excel(writer, sheet_name='Model Data', index=False)

            # --- SHEET 4: Visuals ---
            visuals_sheet = workbook.add_worksheet('Visual Diagnostics')

            # Generate and insert plots in the background
            self._insert_plot(visuals_sheet, 'Actual vs Predicted',
                              self._create_actual_vs_predicted(results, target_col), 'B2')
            self._insert_plot(visuals_sheet, 'Residuals vs Fitted', self._create_residuals_plot(results), 'B25')
            self._insert_plot(visuals_sheet, 'Normal Q-Q', self._create_qq_plot(results), 'O2')

    def export_to_pdf(self, save_path: str, results, target_col: str, diagnostics_dict: dict):
        """
        Exports the regression summary, diagnostics, and charts into a clean PDF tearsheet.
        """
        doc = SimpleDocTemplate(save_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # We will build the PDF by appending elements to this 'story' list
        story = []

        # 1. Title
        title_style = styles['Title']
        story.append(Paragraph(f"Regression Analysis Report", title_style))
        story.append(Paragraph(f"Dependent Variable (Y): {target_col}", styles['Normal']))
        story.append(Spacer(1, 12))

        # 2. Core Metrics Summary Table
        story.append(Paragraph("Model Summary", styles['Heading2']))
        metrics_data = [
            ['Metric', 'Value'],
            ['R-squared', f"{results.rsquared:.4f}"],
            ['Adj. R-squared', f"{results.rsquared_adj:.4f}"],
            ['F-statistic', f"{results.fvalue:.4f}"],
            ['Prob (F-statistic)', f"{results.f_pvalue:.4e}"],
            ['Observations', str(int(results.nobs))]
        ]

        # Style the table to look professional (grey header, grid lines)
        metrics_table = Table(metrics_data, colWidths=[2 * inch, 2 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # 3. Coefficients Table
        story.append(Paragraph("Coefficients", styles['Heading2']))
        coef_data = [['Variable', 'Coefficient', 'Std Error', 't-Stat', 'p-Value']]
        for var_name in results.params.index:
            coef_data.append([
                var_name,
                f"{results.params[var_name]:.4f}",
                f"{results.bse[var_name]:.4f}",
                f"{results.tvalues[var_name]:.4f}",
                f"{results.pvalues[var_name]:.4f}"
            ])

        coef_table = Table(coef_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
        coef_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Keep variable names left-aligned
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(coef_table)
        story.append(Spacer(1, 20))

        # 4. Diagnostics Table
        if diagnostics_dict:
            story.append(Paragraph("Assumption Diagnostics", styles['Heading2']))
            diag_data = [['Test', 'Metric', 'Value']]
            for test_name, metrics in diagnostics_dict.items():
                for key, value in metrics.items():
                    if key != 'Test':
                        diag_data.append([test_name, key, str(value)])

            diag_table = Table(diag_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
            diag_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.slategray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(diag_table)
            story.append(Spacer(1, 20))

        # 5. Visual Diagnostics (Images)
        # We use a page break to ensure the charts don't get awkwardly split across pages
        from reportlab.platypus import PageBreak
        story.append(PageBreak())
        story.append(Paragraph("Visual Diagnostics", styles['Heading1']))

        # Helper to convert a matplotlib figure directly to a ReportLab Image
        def fig_to_rl_image(fig, width, height):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close(fig)
            return RLImage(img_buffer, width=width, height=height)

        # Generate the plots using the existing helper methods
        fig_actual = self._create_actual_vs_predicted(results, target_col)
        fig_resid = self._create_residuals_plot(results)
        fig_qq = self._create_qq_plot(results)

        # Add the images to the story (sized to fit on the page)
        story.append(fig_to_rl_image(fig_actual, 5 * inch, 3.5 * inch))
        story.append(Spacer(1, 10))
        story.append(fig_to_rl_image(fig_resid, 5 * inch, 3.5 * inch))
        story.append(Spacer(1, 10))
        story.append(fig_to_rl_image(fig_qq, 5 * inch, 3.5 * inch))

        # Finally, build the actual PDF file
        doc.build(story)

    # --- Background Plotting Helpers ---
    # We recreate the plots here invisibly so we don't interfere with the PyQt UI canvas

    def _insert_plot(self, worksheet, title, fig, cell_location):
        """Saves a matplotlib figure to an in-memory buffer and inserts it into Excel."""
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
        img_data.seek(0)
        worksheet.write(cell_location.replace('2', '1'), title)  # Add a title right above the image
        worksheet.insert_image(cell_location, '', {'image_data': img_data})
        plt.close(fig)  # Free up memory

    def _create_actual_vs_predicted(self, results, target_name):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("whitegrid")
        y_actual = results.model.endog
        y_predicted = results.fittedvalues
        sns.scatterplot(x=y_actual, y=y_predicted, ax=ax, alpha=0.6, color='#2c3e50')
        mn, mx = y_actual.min(), y_actual.max()
        ax.plot([mn, mx], [mn, mx], color='#e74c3c', linestyle='--')
        ax.set_xlabel(f"Actual {target_name}")
        ax.set_ylabel(f"Predicted")
        return fig

    def _create_residuals_plot(self, results):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("whitegrid")
        sns.scatterplot(x=results.fittedvalues, y=results.resid, ax=ax, alpha=0.6, color='#3498db')
        ax.axhline(0, color='#e74c3c', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        return fig

    def _create_qq_plot(self, results):
        fig, ax = plt.subplots(figsize=(6, 4))
        sm.qqplot(results.resid, line='s', ax=ax, markerfacecolor='#3498db', markeredgecolor='#2980b9', alpha=0.6)
        return fig