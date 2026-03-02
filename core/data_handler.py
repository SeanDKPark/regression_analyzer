import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self):
        """
        Initializes the data handler.
        Stores the current working dataframe to be accessed by the UI or Engine.
        """
        self.df = None

    def load_excel(self, file_path: str, sheet_name: str = 0, cell_range: str = None) -> pd.DataFrame:
        """
        Reads an Excel file, allowing for specific sheet and column/row targeting.

        Parameters:
        - file_path (str): The path to the .xlsx file.
        - sheet_name (str/int): The name or index of the sheet to load.
        - cell_range (str): Excel-style range (e.g., "A:D" for columns, or "A1:D100" for a block).
                            If None, loads the parsed bounds automatically.
        """
        try:
            # If a specific range like "A:F" is provided, we pass it to usecols
            # Pandas handles Excel column letters natively in the usecols parameter
            if cell_range and ":" in cell_range and any(c.isalpha() for c in cell_range):
                # Simple extraction for column letters (e.g., "A:C" -> usecols="A:C")
                # Note: For complex block ranges like "A5:D100", you would parse skiprows and nrows.
                # For simplicity here, we assume column-based selection.
                self.df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=cell_range)
            else:
                self.df = pd.read_excel(file_path, sheet_name=sheet_name)

            return self.df

        except Exception as e:
            raise ValueError(f"Error loading Excel file: {e}")

    def clean_and_prepare(self, df: pd.DataFrame, target_col: str, feature_cols: list,
                          missing_data_strategy: str = 'drop') -> pd.DataFrame:
        """
        Forces selected columns to numeric types and handles missing values.
        This ensures the statsmodels engine doesn't crash on dirty data.
        """
        cols_to_keep = [target_col] + feature_cols

        # 1. Subset the dataframe to only the columns we need for the regression
        clean_df = df[cols_to_keep].copy()

        # 2. Force everything to numeric.
        # 'coerce' turns any accidental strings (like "#N/A" or "Error" from Excel) into pd.NA
        for col in cols_to_keep:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

        # 3. Handle missing data
        if missing_data_strategy == 'drop':
            clean_df = clean_df.dropna()
        elif missing_data_strategy == 'ffill':
            # Forward fill is often standard for time-series market data
            clean_df = clean_df.fillna(method='ffill').dropna()
        elif missing_data_strategy == 'mean':
            clean_df = clean_df.fillna(clean_df.mean())

        return clean_df

    def get_sheet_names(self, file_path: str) -> list:
        """
        Reads the metadata of an Excel file to extract sheet names
        without loading the entire file into memory.
        """
        try:
            # pd.ExcelFile is highly efficient for probing workbook structures
            xls = pd.ExcelFile(file_path)
            return xls.sheet_names
        except Exception as e:
            raise ValueError(f"Error reading sheets: {e}")

    def get_column_names(self) -> list:
        """Returns a list of column names for the UI dropdowns."""
        if self.df is not None:
            return self.df.columns.tolist()
        return []