import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# 1) Load
df = pd.read_csv("data/HR_data.csv")

# 2) Identify numeric vs. categorical columns
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# 3) Scale the numeric block
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[num_cols])

# 4) Reconstruct a mixed DataFrame
df_scaled = pd.DataFrame(scaled_array, columns=num_cols, index=df.index)
df_scaled[cat_cols] = df[cat_cols]

# 5) Plot exactly as before, but on df_scaled
cols = df_scaled.columns.tolist()
n = len(cols)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(5 * ncols, 4 * nrows),
                         squeeze=False)

for ax, col in zip(axes.flatten(), cols):
    series = df_scaled[col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series, bins=30, edgecolor="white")
    else:
        counts = series.value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_title(col)
    ax.set_ylabel("count")

# hide any unused subplots
for ax in axes.flatten()[n:]:
    ax.set_visible(False)

fig.suptitle("Distribution of Every Feature (Numeric Standard-scaled)", fontsize=16, y=1.02)
fig.tight_layout()
plt.show()
