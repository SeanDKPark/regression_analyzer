import pandas as pd
import numpy as np
from core.model_engine import RegressionEngine

# 1. Create some dummy data
np.random.seed(42)
data = {
    'Price': np.random.rand(100) * 100,
    'Interest_Rate': np.random.rand(100) * 5,
    'Volume': np.random.rand(100) * 1000
}
df = pd.DataFrame(data)

# 2. Initialize the engine and run the model
engine = RegressionEngine()
results = engine.run_ols(df, target_col='Price', feature_cols=['Interest_Rate', 'Volume'])

# 3. Print the results
print(engine.get_summary_text(results))