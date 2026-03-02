import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QComboBox, QListWidget, QAbstractItemView, QTextEdit,
                             QMessageBox, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt

# Import core engines
from core.data_handler import DataHandler
from core.model_engine import RegressionEngine
from utils.plotter import AnalysisPlotter
from utils.exporter import ReportExporter


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantitative Regression Analyzer")
        self.resize(1200, 800)

        # Initialize the backend engines
        self.data_handler = DataHandler()
        self.engine = RegressionEngine()

        # State Management
        self.current_results = None
        self.current_target = None
        self.current_file_path = None  # Added to store the path for sheet switching

        # Set up the UI layout
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: Controls ---
        left_panel = QVBoxLayout()

        # 1. File Selection
        self.btn_load = QPushButton("Select Excel File")
        self.btn_load.clicked.connect(self.select_file)  # UPDATED CONNECTION
        self.lbl_file_path = QLabel("No file selected.")
        self.lbl_file_path.setWordWrap(True)

        # --- NEW: Sheet Selection ---
        self.lbl_sheet = QLabel("Select Sheet:")
        self.combo_sheet = QComboBox()
        self.combo_sheet.currentTextChanged.connect(self.load_sheet_data)

        # 2. Variable Selection
        self.lbl_y = QLabel("Select Dependent Variable (Y):")
        self.combo_y = QComboBox()

        self.lbl_x = QLabel("Select Independent Variables (X):")
        self.list_x = QListWidget()
        self.list_x.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        # 3. Model Selection
        self.lbl_model = QLabel("Select Model:")
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Ordinary Least Squares (OLS)"])

        # 4. Run Button
        self.btn_run = QPushButton("Run Regression")
        self.btn_run.clicked.connect(self.execute_regression)

        # --- EXPORT BUTTONS ---
        export_layout = QHBoxLayout()

        self.btn_export_excel = QPushButton("Export to Excel")
        self.btn_export_excel.clicked.connect(self.export_excel)
        self.btn_export_excel.setEnabled(False)

        self.btn_export_pdf = QPushButton("Export to PDF")
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_pdf.setEnabled(False)

        export_layout.addWidget(self.btn_export_excel)
        export_layout.addWidget(self.btn_export_pdf)

        # Add to left panel
        left_panel.addWidget(self.btn_load)
        left_panel.addWidget(self.lbl_file_path)
        left_panel.addSpacing(10)
        left_panel.addWidget(self.lbl_sheet)
        left_panel.addWidget(self.combo_sheet)
        left_panel.addSpacing(20)
        left_panel.addWidget(self.lbl_y)
        left_panel.addWidget(self.combo_y)
        left_panel.addSpacing(10)
        left_panel.addWidget(self.lbl_x)
        left_panel.addWidget(self.list_x)
        left_panel.addSpacing(10)
        left_panel.addWidget(self.lbl_model)
        left_panel.addWidget(self.combo_model)
        left_panel.addStretch()
        left_panel.addWidget(self.btn_run)
        left_panel.addLayout(export_layout)

        # --- RIGHT PANEL: Results (Split View) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top half using QTabWidget
        self.tabs = QTabWidget()

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setFontFamily("Courier")
        self.tabs.addTab(self.text_output, "Statistical Report")

        self.text_diagnostics = QTextEdit()
        self.text_diagnostics.setReadOnly(True)
        self.text_diagnostics.setFontFamily("Courier")
        self.tabs.addTab(self.text_diagnostics, "Assumption Tests")

        # Bottom half: The Plotter Widget with Dropdown
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        plot_header_layout = QHBoxLayout()
        plot_header_layout.addWidget(QLabel("Visual Diagnostics:"))

        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems([
            "Actual vs. Predicted",
            "Residuals vs. Fitted",
            "Normal Q-Q Plot"
        ])
        self.combo_plot_type.currentIndexChanged.connect(self.update_plot)

        plot_header_layout.addWidget(self.combo_plot_type)
        plot_header_layout.addStretch()

        plot_layout.addLayout(plot_header_layout)

        self.plotter = AnalysisPlotter()
        plot_layout.addWidget(self.plotter)

        # Add tabs and plots to the splitter
        splitter.addWidget(self.tabs)
        splitter.addWidget(plot_widget)
        splitter.setSizes([400, 600])

        right_panel_layout.addWidget(splitter)

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(right_panel_widget, 4)

    # --- UI Logic & Connections ---

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self.current_file_path = file_path
            self.lbl_file_path.setText(f"Loaded: {file_path.split('/')[-1]}")
            try:
                self.combo_sheet.blockSignals(True)
                self.combo_sheet.clear()

                sheets = self.data_handler.get_sheet_names(file_path)
                self.combo_sheet.addItems(sheets)

                self.combo_sheet.blockSignals(False)

                if sheets:
                    self.load_sheet_data(sheets[0])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not read Excel file:\n{str(e)}")

    def load_sheet_data(self, sheet_name):
        if not sheet_name or not self.current_file_path:
            return
        try:
            self.data_handler.load_excel(self.current_file_path, sheet_name=sheet_name)
            self.populate_dropdowns()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load sheet data:\n{str(e)}")

    def populate_dropdowns(self):
        self.combo_y.clear()
        self.list_x.clear()
        columns = self.data_handler.get_column_names()
        self.combo_y.addItems(columns)
        self.list_x.addItems(columns)

    def execute_regression(self):
        target_col = self.combo_y.currentText()
        feature_cols = [item.text() for item in self.list_x.selectedItems()]

        if not target_col or not feature_cols:
            QMessageBox.warning(self, "Warning", "Please select both Y and at least one X variable.")
            return

        try:
            clean_df = self.data_handler.clean_and_prepare(self.data_handler.df, target_col, feature_cols)
            model_type = self.combo_model.currentText()

            if model_type == "Ordinary Least Squares (OLS)":
                results = self.engine.run_ols(clean_df, target_col, feature_cols)

                summary_str = self.engine.get_summary_text(results)
                self.text_output.setText(summary_str)

                diagnostics_dict = self.engine.run_diagnostics(results)

                diag_str = "=======================================================\n"
                diag_str += "                 DIAGNOSTIC TESTS\n"
                diag_str += "=======================================================\n\n"

                for test_name, test_data in diagnostics_dict.items():
                    diag_str += f"--- {test_name} ---\n"
                    for key, value in test_data.items():
                        if key != 'Test':
                            diag_str += f"{key:<25}: {value}\n"
                    diag_str += "\n"

                self.text_diagnostics.setText(diag_str)

                self.current_results = results
                self.current_target = target_col
                # We also need to save the clean_df and diagnostics_dict to the class
                # so the export function can grab them later
                self.current_clean_df = clean_df
                self.current_diagnostics = diagnostics_dict

                self.update_plot()
                self.btn_export_excel.setEnabled(True)
                self.btn_export_pdf.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Regression Error", f"An error occurred during calculation:\n{str(e)}")

    def update_plot(self):
        if self.current_results is None:
            return

        selected_plot = self.combo_plot_type.currentText()

        if selected_plot == "Actual vs. Predicted":
            self.plotter.plot_actual_vs_predicted(self.current_results, self.current_target)
        elif selected_plot == "Residuals vs. Fitted":
            self.plotter.plot_residuals_vs_fitted(self.current_results)
        elif selected_plot == "Normal Q-Q Plot":
            self.plotter.plot_qq(self.current_results)

    def export_excel(self):
        if self.current_results is None:
            return

        # 1. Ask the user where to save the file
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Excel Report", "Regression_Report.xlsx",
                                                   "Excel Files (*.xlsx)")

        if save_path:
            try:
                # 2. Call our new exporter
                exporter = ReportExporter()
                exporter.export_to_excel(
                    save_path=save_path,
                    results=self.current_results,
                    target_col=self.current_target,
                    clean_df=self.current_clean_df,
                    diagnostics_dict=self.current_diagnostics
                )
                QMessageBox.information(self, "Success", f"Report successfully exported to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export Excel file:\n{str(e)}")

    def export_pdf(self):
        if self.current_results is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "Regression_Report.pdf",
                                                   "PDF Files (*.pdf)")

        if save_path:
            try:
                exporter = ReportExporter()
                exporter.export_to_pdf(
                    save_path=save_path,
                    results=self.current_results,
                    target_col=self.current_target,
                    diagnostics_dict=self.current_diagnostics
                )
                QMessageBox.information(self, "Success", f"PDF Report successfully exported to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export PDF file:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())