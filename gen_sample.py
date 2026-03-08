import pandas as pd
import numpy as np

# Set seed so the random numbers are exactly the same every time
np.random.seed(42)

# Generate 100 days/months of sample data
# X1: Interest Rate (%) - centered around 3.5%
interest_rate = np.random.normal(loc=3.5, scale=1.0, size=100)

# X2: Market Volatility (VIX proxy) - centered around 18
volatility = np.random.normal(loc=18.0, scale=5.0, size=100)

# X3: GDP Growth (%) - centered around 2.0%
# In FICC, higher growth usually tightens credit spreads
gdp_growth = np.random.normal(loc=2.0, scale=1.5, size=100)

# X4: Inflation Rate (%) - centered around 2.5%
inflation = np.random.normal(loc=2.5, scale=1.0, size=100)

# X5: USD/KRW FX Rate Daily Change (%) - centered around 0%
usd_krw_change = np.random.normal(loc=0.0, scale=0.8, size=100)

# Y: Corporate Bond Spread (bps)
# Base spread of 50 bps
# + widens as rates rise (+15 * X1)
# + widens as volatility spikes (+3 * X2)
# - tightens with strong GDP growth (-8 * X3)
# + slight widening with inflation (+2 * X4)
# + weak relationship with FX changes (+1.5 * X5)
# + random noise
bond_spread = (
    50
    + (15 * interest_rate)
    + (3 * volatility)
    - (8 * gdp_growth)
    + (2 * inflation)
    + (1.5 * usd_krw_change)
    + np.random.normal(loc=0, scale=10, size=100)
)

# Create the DataFrame
df = pd.DataFrame({
    'Bond_Spread_bps': bond_spread,
    'Interest_Rate_pct': interest_rate,
    'Volatility_Index': volatility,
    'GDP_Growth_pct': gdp_growth,
    'Inflation_pct': inflation,
    'USD_KRW_Change_pct': usd_krw_change
})

# Export directly to an Excel file
file_name = "sample_macro_data_5_factors.xlsx"
df.to_excel(file_name, index=False)

print(f"Successfully generated '{file_name}' with {len(df)} rows and 5 independent variables.")