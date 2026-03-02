import pandas as pd
import numpy as np

# Set seed so the random numbers are exactly the same every time
np.random.seed(42)

# Generate 100 days/months of sample data
# X1: Interest Rate (%) - centered around 3.5%
interest_rate = np.random.normal(loc=3.5, scale=1.0, size=100)

# X2: Market Volatility (VIX-like proxy) - centered around 18
volatility = np.random.normal(loc=18.0, scale=5.0, size=100)

# Y: Corporate Bond Spread (bps)
# Base spread of 50 bps + widens as rates rise + widens as volatility spikes + random noise
bond_spread = 50 + (15 * interest_rate) + (3 * volatility) + np.random.normal(loc=0, scale=10, size=100)

# Create the DataFrame
df = pd.DataFrame({
    'Bond_Spread_bps': bond_spread,
    'Interest_Rate_pct': interest_rate,
    'Volatility_Index': volatility
})

# Export directly to an Excel file
file_name = "sample_macro_data.xlsx"
df.to_excel(file_name, index=False)

print(f"Successfully generated '{file_name}' with {len(df)} rows.")