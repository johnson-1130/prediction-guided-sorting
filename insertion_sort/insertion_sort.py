#!/opt/homebrew/bin/python3.12
import sys
sys.path.append('/Users/johnsonchang/Desktop/dissertation materials/sorting with predictions')

import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from positional_prediction import generate_positional_prediction, worst_case
from dirty_comparison import generate_dirty_comparisons

# -------------------------------------------------------
# Common settings
# -------------------------------------------------------

# random.seed(42)

n = 200
NUM_TRIALS = 20
MAX_PAIRS = n * (n - 1) // 2

output_dir = Path(__file__).resolve().parent

error_rates = [
    0.01, 0.02, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
]

# 要標注的 error rate / Points to annotate
dirty_annotate  = {0.01, 0.05, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
pos_annotate    = set(error_rates)  # all

# -------------------------------------------------------
# Insertion Sort
# -------------------------------------------------------

def insertion_sort(arr):
    arr = arr[:]
    swaps = 0
    comparisons = 0
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                swaps += 1
                j -= 1
            else:
                break
        arr[j + 1] = key
    return arr, swaps, comparisons

# -------------------------------------------------------
# O(n log n) inversion count
# -------------------------------------------------------

def merge_count(arr):
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2
    left, lc = merge_count(arr[:mid])
    right, rc = merge_count(arr[mid:])
    merged = []
    count = lc + rc
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            count += len(left) - i
            j += 1
    merged += left[i:]
    merged += right[j:]
    return merged, count

# -------------------------------------------------------
# Positional Prediction (normal distribution)
# -------------------------------------------------------

def generate_positional_prediction_normal(n, error_rate):
    true_ranks = list(range(1, n + 1))
    wc = worst_case(n)
    sigma = (error_rate * wc) / (n * math.sqrt(2 / math.pi))
    p_hat = []
    for i in range(n):
        delta = random.gauss(0, sigma)
        new_rank = true_ranks[i] + round(delta)
        new_rank = max(1, min(n, new_rank))
        p_hat.append(new_rank)
    actual_displacement = sum(abs(p_hat[i] - true_ranks[i]) for i in range(n))
    actual_error_rate = actual_displacement / wc
    return p_hat, actual_error_rate

# -------------------------------------------------------
# Dirty Comparison functions
# -------------------------------------------------------

def out_degree_sort(arr, comparisons):
    out_degree = {x: 0 for x in arr}
    for (x, y), result in comparisons.items():
        if result:
            out_degree[x] += 1
        else:
            out_degree[y] += 1
    return sorted(arr, key=lambda x: out_degree[x], reverse=True)

# -------------------------------------------------------
# Annotation helper (alternating up/down)
# -------------------------------------------------------

def annotate_scatter(ax, x_vals, y_vals, target_rates, annotate_set,
                     color, fontsize=6, force_above=None):
    if force_above is None:
        force_above = set()
    annotated_idx = 0
    for xi, yi, target in zip(x_vals, y_vals, target_rates):
        if target not in annotate_set:
            continue
        if target in force_above:
            yoff = 10
            va = 'bottom'
        elif annotated_idx % 2 == 0:
            yoff = 10
            va = 'bottom'
        else:
            yoff = -10
            va = 'top'
        # target 越小 zorder 越高，讓小 target 的標注蓋在大 target 上面
        # Smaller target gets higher zorder so it appears on top
        zorder = 100 - int(target * 100)
        ax.annotate(
            f'target {target*100:.0f}%\ninv {xi:.1f}%',
            xy=(xi, yi),
            xytext=(0, yoff),
            textcoords='offset points',
            ha='center', va=va, fontsize=fontsize, color=color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec=color, lw=0.6, alpha=0.8),
            zorder=zorder
        )
        annotated_idx += 1

# -------------------------------------------------------
# Experiment: Positional Prediction + Insertion Sort
# -------------------------------------------------------

pos_inv_ratios = []
pos_avg_swaps  = []
pos_avg_comps  = []

for error_rate in error_rates:
    trial_inv   = []
    trial_swaps = []
    trial_comps = []

    for _ in range(NUM_TRIALS):
        p_hat, _ = generate_positional_prediction_normal(n, error_rate)
        _, inv = merge_count(p_hat)
        trial_inv.append(inv / MAX_PAIRS)
        _, swaps, comps = insertion_sort(p_hat)
        trial_swaps.append(swaps)
        trial_comps.append(comps)

    pos_inv_ratios.append(sum(trial_inv) / NUM_TRIALS)
    pos_avg_swaps.append(sum(trial_swaps) / NUM_TRIALS)
    pos_avg_comps.append(sum(trial_comps) / NUM_TRIALS)
    print(f"[Positional] target={error_rate:.2%}  "
          f"avg_inv={pos_inv_ratios[-1]*100:.2f}%  "
          f"avg_swaps={pos_avg_swaps[-1]:.1f}  "
          f"avg_comps={pos_avg_comps[-1]:.1f}")

# -------------------------------------------------------
# Experiment: Dirty Comparison + Insertion Sort
# -------------------------------------------------------

arr        = list(range(1, n + 1))
true_order = sorted(arr)

dirty_inv_ratios = []
dirty_avg_swaps  = []
dirty_avg_comps  = []

for error_rate in error_rates:
    trial_inv   = []
    trial_swaps = []
    trial_comps = []

    for _ in range(NUM_TRIALS):
        comparisons, _ = generate_dirty_comparisons(arr, error_rate)
        predicted_order = out_degree_sort(arr, comparisons)
        true_rank  = {v: i for i, v in enumerate(true_order)}
        rank_list  = [true_rank[x] for x in predicted_order]
        _, inv     = merge_count(rank_list)
        trial_inv.append(inv / MAX_PAIRS)
        _, swaps, comps = insertion_sort(predicted_order)
        trial_swaps.append(swaps)
        trial_comps.append(comps)

    dirty_inv_ratios.append(sum(trial_inv) / NUM_TRIALS)
    dirty_avg_swaps.append(sum(trial_swaps) / NUM_TRIALS)
    dirty_avg_comps.append(sum(trial_comps) / NUM_TRIALS)
    print(f"[Dirty]      target={error_rate:.2%}  "
          f"avg_inv={dirty_inv_ratios[-1]*100:.2f}%  "
          f"avg_swaps={dirty_avg_swaps[-1]:.1f}  "
          f"avg_comps={dirty_avg_comps[-1]:.1f}")

pos_x   = [r * 100 for r in pos_inv_ratios]
dirty_x = [r * 100 for r in dirty_inv_ratios]

# -------------------------------------------------------
# Plot 1: Positional — Swaps vs Inversion Ratio
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(pos_x, pos_avg_swaps,
        marker='o', color='steelblue', linewidth=2, markersize=5, label='Swaps')
theory_swaps = [r * MAX_PAIRS for r in pos_inv_ratios]
ax.plot(pos_x, theory_swaps,
        linestyle='--', color='black', alpha=0.4, linewidth=1.2, label='Theory: Inv')

annotate_scatter(ax, pos_x, pos_avg_swaps, error_rates,
                 pos_annotate, color='steelblue', fontsize=6)

ax.set_xlabel('Inversion Ratio (%)')
ax.set_ylabel('Number of Swaps')
ax.set_title(f'Positional Prediction: Insertion Sort Swaps vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'ins_positional_swaps.png', dpi=150)
plt.close()
print("Plot 1 saved: ins_positional_swaps.png\n")

# -------------------------------------------------------
# Plot 2: Positional — Comparisons vs Inversion Ratio
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(pos_x, pos_avg_comps,
        marker='o', color='steelblue', linewidth=2, markersize=5, label='Comparisons')
theory_comps = [n + r * MAX_PAIRS for r in pos_inv_ratios]
ax.plot(pos_x, theory_comps,
        linestyle='--', color='black', alpha=0.4, linewidth=1.2, label='Theory: n + Inv')

annotate_scatter(ax, pos_x, pos_avg_comps, error_rates,
                 pos_annotate, color='steelblue', fontsize=6)

ax.set_xlabel('Inversion Ratio (%)')
ax.set_ylabel('Number of Comparisons')
ax.set_title(f'Positional Prediction: Insertion Sort Comparisons vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'ins_positional_comps.png', dpi=150)
plt.close()
print("Plot 2 saved: ins_positional_comps.png\n")

# -------------------------------------------------------
# Plot 3: Dirty — Swaps vs Inversion Ratio
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dirty_x, dirty_avg_swaps,
        marker='s', color='tomato', linewidth=2, markersize=5, label='Swaps')
theory_swaps_dirty = [r * MAX_PAIRS for r in dirty_inv_ratios]
ax.plot(dirty_x, theory_swaps_dirty,
        linestyle='--', color='black', alpha=0.4, linewidth=1.2, label='Theory: Inv')

annotate_scatter(ax, dirty_x, dirty_avg_swaps, error_rates,
                 dirty_annotate, color='tomato', fontsize=6)

ax.set_xlabel('Inversion Ratio (%)')
ax.set_ylabel('Number of Swaps')
ax.set_title(f'Dirty Comparison: Insertion Sort Swaps vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'ins_dirty_swaps.png', dpi=150)
plt.close()
print("Plot 3 saved: ins_dirty_swaps.png\n")

# -------------------------------------------------------
# Plot 4: Dirty — Comparisons vs Inversion Ratio
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dirty_x, dirty_avg_comps,
        marker='s', color='tomato', linewidth=2, markersize=5, label='Comparisons')
theory_comps_dirty = [n + r * MAX_PAIRS for r in dirty_inv_ratios]
ax.plot(dirty_x, theory_comps_dirty,
        linestyle='--', color='black', alpha=0.4, linewidth=1.2, label='Theory: n + Inv')

annotate_scatter(ax, dirty_x, dirty_avg_comps, error_rates,
                 dirty_annotate, color='tomato', fontsize=6)

ax.set_xlabel('Inversion Ratio (%)')
ax.set_ylabel('Number of Comparisons')
ax.set_title(f'Dirty Comparison: Insertion Sort Comparisons vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'ins_dirty_comps.png', dpi=150)
plt.close()
print("Plot 4 saved: ins_dirty_comps.png")