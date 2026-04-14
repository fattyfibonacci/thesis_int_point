import numpy as np
import pandas as pd
import copy as copy
import scipy
import scipy.io
import time
import os
from scipy.linalg import solve, LinAlgWarning
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random

from matplotlib.animation import FuncAnimation, PillowWriter
import openpyxl
import xlsxwriter

from old_files.inpoint_methods import paso_intpoint, loadProblem, intpoint, intpointR, highlight_greaterthan #,intpointR_mask

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def analyze_components(df, row_start=1, row_end=-1, std_threshold=1, sort_within_groups=True):
    """
    Analyze component-wise changes in a dataframe across iterations.

    Parameters:
    - df: pandas DataFrame (rows = iterations, columns = components)
    - row_start: row index to compare from (default: 1)
    - row_end: row index to compare to (default: last row)
    - std_threshold: number of std deviations for 'non-extreme' (default: 1)
    - sort_within_groups: whether to sort columns by difference (default: True)

    Returns:
    - dict of grouped DataFrames
    """

    # --- Step 1: differences ---
    diffs = df.iloc[row_end] - df.iloc[row_start]

    mean = diffs.mean()
    std = diffs.std()

    # --- Step 2: masks ---
    inc_mask = diffs > 0
    dec_mask = diffs < 0

    within_std_mask = (diffs >= mean - std_threshold * std) & \
                      (diffs <= mean + std_threshold * std)

    extreme_mask = ~within_std_mask

    # --- Step 3: column groups ---
    inc_normal_cols = diffs[inc_mask & within_std_mask].index
    dec_normal_cols = diffs[dec_mask & within_std_mask].index
    inc_extreme_cols = diffs[inc_mask & extreme_mask].index
    dec_extreme_cols = diffs[dec_mask & extreme_mask].index

    # --- Step 4: build DataFrames ---
    groups = {
        "inc_normal": df[inc_normal_cols],
        "dec_normal": df[dec_normal_cols],
        "inc_extreme": df[inc_extreme_cols],
        "dec_extreme": df[dec_extreme_cols],
    }

    # --- Step 5: optional sorting ---
    if sort_within_groups:
        for key, gdf in groups.items():
            cols = gdf.columns
            groups[key] = gdf[diffs[cols].sort_values().index]

    # --- Step 6: print summary ---
    print(f"inc_normal: {len(inc_normal_cols)}")
    print(f"dec_normal: {len(dec_normal_cols)}")
    print(f"inc_extreme: {len(inc_extreme_cols)}")
    print(f"dec_extreme: {len(dec_extreme_cols)}")

    # --- Step 7: plotting helper ---
    def plot_dots(gdf, title):
        if gdf.shape[1] == 0:
            print(f"{title}: no data")
            return

        plt.figure(figsize=(12, 6))

        n_cols = gdf.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_cols))

        for i, col in enumerate(gdf.columns):
            plt.scatter(gdf.index, gdf[col],
                        color=colors[i],
                        s=8,
                        alpha=0.4)

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # --- Step 8: plots ---
    plot_dots(groups["inc_normal"], "Increasing Components (Non-Extreme)")
    plot_dots(groups["dec_normal"], "Decreasing Components (Non-Extreme)")
    plot_dots(groups["inc_extreme"], "Increasing Components (Extreme)")
    plot_dots(groups["dec_extreme"], "Decreasing Components (Extreme)")

    return groups


def graph_selected_components_trajectory(muzdf,num_columns_to_plot,num_iterations,seed=42,indexes=None):
# Graphs the trajectory of the last "x" number of iterations of any amount of components
    if indexes==None:
        random.seed(seed)
        indexes = random.sample(range(0, muzdf.shape[1]), num_columns_to_plot)

    random_muzdf = muzdf.iloc[-num_iterations:, indexes]  # only the subset

    # -----------------------------
    # Plot dots
    # -----------------------------
    plt.figure(figsize=(12,6))

    for col in random_muzdf.columns:
        plt.plot(random_muzdf.index, random_muzdf[col], 'o', label=f'Component {col}', markersize=8)

    plt.xlabel("Iteration")
    plt.ylabel("z / mu")
    plt.title(f"Behavior of selected components over last {num_iterations} iterations (dots)")
    plt.legend()
    plt.grid(True)
    plt.show()

def graph_mu_vs_z_one_component_static(mudf,zdf,sample_column):

    # Take the first column
    x = mudf.iloc[:, 0]  # mudf values
    y = zdf.iloc[:, 0]    # zdf values

    # Static scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='blue')
    plt.xlabel("Mu (multiplicador de Lagrange)")
    plt.ylabel("z (holgura)")
    plt.title(f"Mu vs z - column {sample_column} over iterations")
    plt.grid(True)
    plt.show()
    


def find_components_tending_to_zero(mudf=None, zdf=None, 
                                   threshold=1e-4, 
                                   last_n_iters=5):
    """
    Detect components tending to zero over the last iterations.

    Usage:
    - mudf only  → checks mu
    - zdf only   → checks z
    - both       → checks mu AND z

    Criteria:
    - |value| < threshold
    - Must hold for ALL of the last iterations

    Returns:
    - list of column indices
    """

    if mudf is None and zdf is None:
        raise ValueError("At least one of mudf or zdf must be provided")

    # --- Step 1: slice last iterations ---
    if mudf is not None:
        mu_tail = mudf.iloc[-last_n_iters:]
        mu_small = mu_tail.abs() < threshold

    if zdf is not None:
        z_tail = zdf.iloc[-last_n_iters:]
        z_small = z_tail.abs() < threshold

    # --- Step 2: combine logic ---
    if mudf is not None and zdf is not None:
        valid = (mu_small & z_small).all(axis=0)
        label = "mu AND z"
    elif mudf is not None:
        valid = mu_small.all(axis=0)
        label = "mu only"
    else:
        valid = z_small.all(axis=0)
        label = "z only"

    selected = valid[valid].index.tolist()

    print(f"Components where {label} → 0 "
          f"(threshold={threshold}, last {last_n_iters} iters): {len(selected)}")
    
    print(f"indexes selected = {selected}")
    return selected