import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from pathlib import Path

# Path-safe and OS-agnostic
csv_data = Path("compressed_results_500kb") / "train_images" / "D36" / "D36_metrics.csv"
df = pd.read_csv(csv_data)

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["PSNR", "SSIM", "BRISQUE"]

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    ax.set_title(f"{metric} by Image and Format")
    
    for fmt in df["Format"].unique():
        subset = df[df["Format"] == fmt]
        
        # Drop rows with NaNs in the current metric
        subset = subset.dropna(subset=[metric])
        
        if subset.empty:
            continue
        
        x = np.arange(len(subset))
        y = subset[metric].values

        if len(x) >= 4:  # Spline needs at least 4 points
            try:
                spline = make_interp_spline(x, y)
                x_smooth = np.linspace(x.min(), x.max(), 200)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, label=fmt)
            except ValueError:
                # In case cleaned data still has bad values
                ax.plot(x, y, label=fmt)
        else:
            ax.plot(x, y, label=fmt)

    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(subset["Image"], rotation=45)
    ax.legend()

plt.tight_layout()
plt.show()