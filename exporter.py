import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, \
    Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class ReportExporter:
    def __init__(self):
        pass

    def export_to_excel(self, save_path: str, results, target_col: str, feature_cols: list, clean_df: pd.DataFrame,
                        diagnostics_dict: dict, metrics_df: pd.DataFrame, full_anova_str: str, subset_anovas_str: str):
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- SHEET 1: Model Summary ---
            summary_stats = pd.DataFrame({
                'Metric': ['Dependent Variable (Y)', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
                           'No. Observations'],
                'Value': [target_col, results.rsquared, results.rsquared_adj, results.fvalue, results.f_pvalue,
                          int(results.nobs)]
            })
            summary_stats.to_excel(writer, sheet_name='Model Summary', index=False, startrow=0, startcol=0)

            coef_df = pd.DataFrame({
                'Coefficient': results.params,
                'Std Error': results.bse,
                't-Statistic': results.tvalues,
                'p-Value': results.pvalues
            })
            coef_df.index.name = 'Variable'
            coef_df.to_excel(writer, sheet_name='Model Summary', startrow=len(summary_stats) + 2)

            # --- SHEET 2: ANOVAs (New) ---
            anova_sheet = workbook.add_worksheet('ANOVA Tables')
            # Removed text_wrap so Excel doesn't struggle to render height, just standard monospace
            format_mono = workbook.add_format({'font_name': 'Courier New'})
            anova_sheet.set_column('A:A', 100)

            combined_anova_text = f"--- FULL MODEL ANOVA ---\n\n{full_anova_str}\n\n\n--- ALL SUBSET ANOVAS ---\n\n{subset_anovas_str}"

            # Split the massive string by line breaks and write each line to a new row
            for row_idx, line in enumerate(combined_anova_text.split('\n')):
                # write_string forces Excel to ignore the "=" signs and treat it as text
                anova_sheet.write_string(row_idx, 0, line, format_mono)

            # --- SHEET 3: Diagnostics ---
            diag_rows = []
            if diagnostics_dict:
                for test_name, metrics in diagnostics_dict.items():
                    for key, value in metrics.items():
                        if key != 'Test':
                            diag_rows.append({'Assumption Test': test_name, 'Metric': key, 'Value': value})
            pd.DataFrame(diag_rows).to_excel(writer, sheet_name='Diagnostics', index=False)

            # --- SHEET 4: Subset Metrics (Highlighted) ---
            if metrics_df is not None:
                # Helper function to apply CSS styling to pandas
                def highlight_best(col):
                    if col.name == 'AIC':
                        return ['background-color: #cce5ff; font-weight: bold' if v else '' for v in (col == col.min())]
                    elif col.name == 'BIC':
                        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in (col == col.min())]
                    elif col.name == 'Adj_R2':
                        return ['background-color: #fff3cd; font-weight: bold' if v else '' for v in (col == col.max())]
                    return [''] * len(col)

                styled_df = metrics_df.style.apply(highlight_best, axis=0)
                styled_df.to_excel(writer, sheet_name='Subset Metrics', index=False)

            # --- SHEET 5: Model Data ---
            output_df = clean_df.copy()
            output_df['Predicted_Y'] = results.fittedvalues
            output_df['Residuals'] = results.resid
            output_df.to_excel(writer, sheet_name='Model Data', index=False)

            # --- SHEET 6: Visuals ---
            visuals_sheet = workbook.add_worksheet('Visual Diagnostics')
            self._insert_plot(visuals_sheet, 'Actual vs Predicted',
                              self._create_actual_vs_predicted(results, target_col), 'B2')
            self._insert_plot(visuals_sheet, 'Normal Q-Q', self._create_qq_plot(results), 'O2')
            self._insert_plot(visuals_sheet, 'Subset Selection Metrics',
                              self._create_subset_chart(metrics_df, len(feature_cols)), 'B25')
            self._insert_plot(visuals_sheet, 'All Residuals', self._create_all_residuals(results, feature_cols), 'O25')
            self._insert_plot(visuals_sheet, 'Factor Pairplot',
                              self._create_pairplot(clean_df, target_col, feature_cols), 'B55')

    def export_to_pdf(self, save_path: str, results, target_col: str, feature_cols: list, clean_df: pd.DataFrame,
                      diagnostics_dict: dict, metrics_df: pd.DataFrame, full_anova_str: str, subset_anovas_str: str):
        doc = SimpleDocTemplate(save_path, pagesize=letter)
        styles = getSampleStyleSheet()
        mono_style = ParagraphStyle('Courier', parent=styles['Normal'], fontName='Courier', fontSize=8, leading=10)
        story = []

        # 1. Title
        story.append(Paragraph("Quantitative Regression Report", styles['Title']))
        story.append(Paragraph(f"Target Variable (Y): {target_col}", styles['Normal']))
        story.append(Spacer(1, 12))

        # 2. Core Metrics
        story.append(Paragraph("Full Model Summary", styles['Heading2']))
        metrics_data = [['Metric', 'Value'], ['R-squared', f"{results.rsquared:.4f}"],
                        ['Adj. R-squared', f"{results.rsquared_adj:.4f}"],
                        ['F-statistic', f"{results.fvalue:.4f}"], ['Prob (F-statistic)', f"{results.f_pvalue:.4e}"],
                        ['Observations', str(int(results.nobs))]]

        metrics_table = Table(metrics_data, colWidths=[2 * inch, 2 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # 3. Coefficients
        story.append(Paragraph("Coefficients", styles['Heading2']))
        coef_data = [['Variable', 'Coefficient', 'Std Error', 't-Stat', 'p-Value']]
        for var_name in results.params.index:
            coef_data.append([var_name, f"{results.params[var_name]:.4f}", f"{results.bse[var_name]:.4f}",
                              f"{results.tvalues[var_name]:.4f}", f"{results.pvalues[var_name]:.4f}"])

        coef_table = Table(coef_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
        coef_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'), ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(coef_table)
        story.append(Spacer(1, 20))

        # 4. Subset Metrics Table (ALL rows, highlighted)
        if metrics_df is not None:
            story.append(PageBreak())
            story.append(Paragraph("All Subset Models", styles['Heading2']))

            subset_data = [['Model', 'AIC', 'BIC', 'R-squared', 'Adj R-squared']]
            best_aic, best_bic, best_adj_r2 = metrics_df['AIC'].min(), metrics_df['BIC'].min(), metrics_df[
                'Adj_R2'].max()

            table_styles = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkslategray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]

            for r_idx, row in metrics_df.iterrows():
                row_num = r_idx + 1  # Offset for header
                subset_data.append([str(row['Model']), f"{row['AIC']:.4f}", f"{row['BIC']:.4f}", f"{row['R2']:.4f}",
                                    f"{row['Adj_R2']:.4f}"])

                # Dynamically apply highlighting to the correct cell coordinates
                if row['AIC'] == best_aic:
                    table_styles.extend([('BACKGROUND', (1, row_num), (1, row_num), colors.HexColor('#cce5ff')),
                                         ('FONTNAME', (1, row_num), (1, row_num), 'Helvetica-Bold')])
                if row['BIC'] == best_bic:
                    table_styles.extend([('BACKGROUND', (2, row_num), (2, row_num), colors.HexColor('#d4edda')),
                                         ('FONTNAME', (2, row_num), (2, row_num), 'Helvetica-Bold')])
                if row['Adj_R2'] == best_adj_r2:
                    table_styles.extend([('BACKGROUND', (4, row_num), (4, row_num), colors.HexColor('#fff3cd')),
                                         ('FONTNAME', (4, row_num), (4, row_num), 'Helvetica-Bold')])

            subset_table = Table(subset_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
            subset_table.setStyle(TableStyle(table_styles))
            story.append(subset_table)

        # 5. Full ANOVA
        story.append(PageBreak())
        story.append(Paragraph("Full Model ANOVA", styles['Heading2']))
        story.append(Preformatted(full_anova_str, mono_style))

        # 6. Visual Diagnostics
        story.append(PageBreak())
        story.append(Paragraph("Visual Diagnostics", styles['Heading1']))

        def fig_to_rl_image(fig, width, height):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            plt.close(fig)
            return RLImage(img_buffer, width=width, height=height)

        fig_subset = self._create_subset_chart(metrics_df, len(feature_cols))
        story.append(fig_to_rl_image(fig_subset, 6.5 * inch, 4 * inch))
        story.append(Spacer(1, 15))

        fig_actual = self._create_actual_vs_predicted(results, target_col)
        story.append(fig_to_rl_image(fig_actual, 5 * inch, 3.5 * inch))

        story.append(PageBreak())
        fig_resid = self._create_all_residuals(results, feature_cols)
        resid_h = min(8.5, 2.5 * (len(feature_cols) + 1))
        story.append(fig_to_rl_image(fig_resid, 5 * inch, resid_h * inch))

        # 7. Subset ANOVAs (At the very end because it's long)
        story.append(PageBreak())
        story.append(Paragraph("All Subset ANOVAs", styles['Heading2']))
        story.append(Preformatted(subset_anovas_str, mono_style))

        doc.build(story)

    # --- Background Plotting Helpers ---
    def _insert_plot(self, worksheet, title, fig, cell_location):
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight', dpi=100)
        img_data.seek(0)
        worksheet.write(cell_location.replace('2', '1').replace('5', '4'), title)
        worksheet.insert_image(cell_location, '', {'image_data': img_data})
        plt.close(fig)

    def _create_actual_vs_predicted(self, results, target_name):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("whitegrid")
        y_actual, y_predicted = results.model.endog, results.fittedvalues
        sns.scatterplot(x=y_actual, y=y_predicted, ax=ax, alpha=0.6, color='#2c3e50')
        mn, mx = y_actual.min(), y_actual.max()
        ax.plot([mn, mx], [mn, mx], color='#e74c3c', linestyle='--')
        ax.set_xlabel(f"Actual {target_name}")
        ax.set_ylabel("Predicted")
        return fig

    def _create_qq_plot(self, results):
        fig, ax = plt.subplots(figsize=(6, 4))
        sm.qqplot(results.resid, line='s', ax=ax, markerfacecolor='#3498db', markeredgecolor='#2980b9', alpha=0.6)
        return fig

    def _create_all_residuals(self, results, feature_names):
        k = len(feature_names)
        fig, axes = plt.subplots(nrows=k + 1, ncols=1, figsize=(6, 3 * (k + 1)), tight_layout=True)
        if k == 0: axes = [axes]
        sns.scatterplot(x=results.fittedvalues, y=results.resid, ax=axes[0], alpha=0.6, color='#3498db')
        axes[0].axhline(0, color='#e74c3c', linestyle='--')
        axes[0].set_title("Residuals vs. Fitted")
        exog_df = results.model.data.orig_exog
        for i, feature in enumerate(feature_names):
            sns.scatterplot(x=exog_df[feature], y=results.resid, ax=axes[i + 1], alpha=0.6, color='#9b59b6')
            axes[i + 1].axhline(0, color='#e74c3c', linestyle='--')
            axes[i + 1].set_title(f"Residuals vs. {feature}")
        return fig

    def _create_pairplot(self, df, target_name, feature_names):
        plot_df = df[feature_names + [target_name]]
        g = sns.pairplot(plot_df, corner=True, kind='reg',
                         plot_kws={'line_kws': {'color': '#e74c3c'}, 'scatter_kws': {'alpha': 0.6, 'color': '#16a085'}})
        return g.fig

    def _create_subset_chart(self, metrics_df, total_features):
        df_sorted = metrics_df.sort_values(by='BIC').reset_index(drop=True)
        fig, ax1 = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax1.plot(df_sorted['Model'], df_sorted['AIC'], color='#00BFFF', lw=2, label='AIC')
        ax1.plot(df_sorted['Model'], df_sorted['BIC'], color='#0047AB', lw=2, label='BIC')
        ax1.tick_params(axis='x', rotation=90, labelsize=7)
        ax2 = ax1.twinx()
        ax2.plot(df_sorted['Model'], df_sorted['R2'], color='#00FA9A', lw=2, label='R2')
        ax2.plot(df_sorted['Model'], df_sorted['Adj_R2'], color='#DC143C', lw=2, label='Adjusted R2')
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        return fig