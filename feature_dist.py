import matplotlib.pyplot as plt
import pandas as pd
import math


# df = pd.read_csv("your_file.csv")
df = pd.read_csv("data/HR_data.csv") 

cols = df.columns.tolist()
n = len(cols)
ncols = 3                        # how many plots per row
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(5 * ncols, 4 * nrows),
                         squeeze=False)

for ax, col in zip(axes.flatten(), cols):
    series = df[col].dropna()
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

fig.suptitle("Distribution of Every Feature", fontsize=16, y=1.02)
fig.tight_layout()
plt.show()