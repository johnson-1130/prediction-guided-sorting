#!/opt/homebrew/bin/python3.12
import sys
sys.path.append('/Users/johnsonchang/Desktop/dissertation materials/sorting with predictions')

import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from positional_prediction import worst_case
from dirty_comparison import generate_dirty_comparisons

# -------------------------------------------------------
# Common settings
# -------------------------------------------------------

# random.seed(42)

n = 600
NUM_TRIALS = 20
MAX_PAIRS = n * (n - 1) // 2

output_dir = Path(__file__).resolve().parent

error_rates = [
    0.01, 0.02, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
]

low_error_rates = [0.01, 0.02, 0.05, 0.10]

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
# Standard deviation
# -------------------------------------------------------

def std_dev(trials):
    avg = sum(trials) / len(trials)
    return math.sqrt(sum((x - avg) ** 2 for x in trials) / len(trials))

# -------------------------------------------------------
# Common annotation function
# -------------------------------------------------------

def annotate_points(ax, x_vals, avg_vals, max_vals, min_vals,
                    color, std_vals=None, fontsize=6, offset=12):
    for idx, (xi, avg, hi, lo) in enumerate(
            zip(x_vals, avg_vals, max_vals, min_vals)):

        if idx == 0:
            ha = 'right'
            xoff = -4
        else:
            ha = 'center'
            xoff = 0

        ax.annotate(
            f'max {hi*100:.1f}%\nmin {lo*100:.1f}%',
            xy=(xi, avg),
            xytext=(xoff, offset + 14),
            textcoords='offset points',
            ha=ha, fontsize=fontsize, color=color
        )

        std_str = f'\nstd {std_vals[idx]*100:.2f}%' if std_vals is not None else ''
        ax.annotate(
            f'avg {avg*100:.1f}%{std_str}',
            xy=(xi, avg),
            xytext=(xoff, offset),
            textcoords='offset points',
            ha=ha, fontsize=fontsize, color=color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec=color, lw=0.8, alpha=0.9)
        )

        ax.annotate(
            f'target {xi:.0f}%',
            xy=(xi, avg),
            xytext=(xoff, -offset - 4),
            textcoords='offset points',
            ha=ha, fontsize=fontsize, color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='black', lw=0.8, alpha=0.9)
        )

# -------------------------------------------------------
# Plot 1: Positional Prediction (1%-50%)
# -------------------------------------------------------

positional_all_trials = []

for error_rate in error_rates:
    trial_ratios = []
    for _ in range(NUM_TRIALS):
        p_hat, _ = generate_positional_prediction_normal(n, error_rate)
        _, inv = merge_count(p_hat)
        trial_ratios.append(inv / MAX_PAIRS)
    positional_all_trials.append(trial_ratios)
    avg = sum(trial_ratios) / NUM_TRIALS
    print(f"[Positional] target={error_rate:.2%}  "
          f"avg={avg*100:.2f}%  "
          f"min={min(trial_ratios)*100:.2f}%  "
          f"max={max(trial_ratios)*100:.2f}%  "
          f"std={std_dev(trial_ratios)*100:.2f}%")

x = [r * 100 for r in error_rates]

avg_ratios = [sum(t) / NUM_TRIALS for t in positional_all_trials]
max_ratios = [max(t) for t in positional_all_trials]
min_ratios = [min(t) for t in positional_all_trials]
std_ratios = [std_dev(t) for t in positional_all_trials]

fig, ax = plt.subplots(figsize=(9, 5))

for trial_idx in range(NUM_TRIALS):
    y = [positional_all_trials[er_idx][trial_idx]
         for er_idx in range(len(error_rates))]
    ax.plot(x, y, color='steelblue', alpha=0.08, linewidth=0.8)

ax.plot(x, avg_ratios, color='steelblue', linewidth=2.5,
        marker='o', markersize=5, label='Average', zorder=5)

ax.errorbar(x, avg_ratios,
            yerr=std_ratios,
            fmt='none', color='steelblue', alpha=0.5,
            capsize=4, linewidth=1.2, zorder=4)

annotate_points(ax, x, avg_ratios, max_ratios, min_ratios,
                color='steelblue', std_vals=std_ratios, fontsize=6, offset=12)

ax.set_xlabel('Target Error Parameter (%)')
ax.set_ylabel('Inversion Ratio (inversions / C(n,2))')
ax.set_title(f'Positional Prediction: Error Rate vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials, normal distribution)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'avg_positional_1_50.png', dpi=150)
plt.close()
print("Plot 1 saved: avg_positional_1_50.png\n")

# -------------------------------------------------------
# Plot 2: Positional Prediction (1%-10%)
# -------------------------------------------------------

low_idx = [error_rates.index(r) for r in low_error_rates]
low_x   = [r * 100 for r in low_error_rates]

low_avg = [avg_ratios[i] for i in low_idx]
low_max = [max_ratios[i] for i in low_idx]
low_min = [min_ratios[i] for i in low_idx]
low_std = [std_ratios[i] for i in low_idx]

fig, ax = plt.subplots(figsize=(8, 5))

for trial_idx in range(NUM_TRIALS):
    y = [positional_all_trials[i][trial_idx] for i in low_idx]
    ax.plot(low_x, y, color='steelblue', alpha=0.08, linewidth=0.8)

ax.plot(low_x, low_avg, color='steelblue', linewidth=2.5,
        marker='o', markersize=6, label='Average', zorder=5)

ax.errorbar(low_x, low_avg,
            yerr=low_std,
            fmt='none', color='steelblue', alpha=0.5,
            capsize=4, linewidth=1.2, zorder=4)

annotate_points(ax, low_x, low_avg, low_max, low_min,
                color='steelblue', std_vals=low_std, fontsize=7, offset=12)

ax.set_xticks(low_x)
ax.set_xticklabels(['1%', '2%', '5%', '10%'])
ax.set_xlabel('Target Error Parameter (%)')
ax.set_ylabel('Inversion Ratio (inversions / C(n,2))')
ax.set_title(f'Positional Prediction: Low Error Rates\n'
             f'(n={n}, {NUM_TRIALS} trials, normal distribution)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'avg_positional_1_10.png', dpi=150)
plt.close()
print("Plot 2 saved: avg_positional_1_10.png\n")

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


def inversions_from_order(sorted_arr, true_order):
    true_rank = {v: i for i, v in enumerate(true_order)}
    rank_list = [true_rank[x] for x in sorted_arr]
    _, count = merge_count(rank_list)
    return count

# -------------------------------------------------------
# Plot 3: Dirty Comparison (1%-50%)
# -------------------------------------------------------

arr        = list(range(1, n + 1))
true_order = sorted(arr)

dirty_all_trials = []

for error_rate in error_rates:
    trial_ratios = []
    for _ in range(NUM_TRIALS):
        comparisons, _ = generate_dirty_comparisons(arr, error_rate)
        predicted_order = out_degree_sort(arr, comparisons)
        inv = inversions_from_order(predicted_order, true_order)
        trial_ratios.append(inv / MAX_PAIRS)
    dirty_all_trials.append(trial_ratios)
    avg = sum(trial_ratios) / NUM_TRIALS
    print(f"[Dirty]      target={error_rate:.2%}  "
          f"avg={avg*100:.2f}%  "
          f"min={min(trial_ratios)*100:.2f}%  "
          f"max={max(trial_ratios)*100:.2f}%  "
          f"std={std_dev(trial_ratios)*100:.2f}%")

dirty_avg = [sum(t) / NUM_TRIALS for t in dirty_all_trials]
dirty_max = [max(t) for t in dirty_all_trials]
dirty_min = [min(t) for t in dirty_all_trials]
dirty_std = [std_dev(t) for t in dirty_all_trials]

fig, ax = plt.subplots(figsize=(9, 5))

for trial_idx in range(NUM_TRIALS):
    y = [dirty_all_trials[er_idx][trial_idx]
         for er_idx in range(len(error_rates))]
    ax.plot(x, y, color='tomato', alpha=0.08, linewidth=0.8)

ax.plot(x, dirty_avg, color='tomato', linewidth=2.5,
        marker='s', markersize=5, label='Average', zorder=5)

ax.errorbar(x, dirty_avg,
            yerr=dirty_std,
            fmt='none', color='tomato', alpha=0.5,
            capsize=4, linewidth=1.2, zorder=4)

annotate_points(ax, x, dirty_avg, dirty_max, dirty_min,
                color='tomato', std_vals=dirty_std, fontsize=6, offset=12)

ax.set_xlabel('Dirty Comparison Error Rate (%)')
ax.set_ylabel('Inversion Ratio (inversions / C(n,2))')
ax.set_title(f'Dirty Comparison: Error Rate vs Inversion Ratio\n'
             f'(n={n}, {NUM_TRIALS} trials each)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'avg_dirty_1_50.png', dpi=150)
plt.close()
print("Plot 3 saved: avg_dirty_1_50.png\n")

# -------------------------------------------------------
# Plot 4: Dirty Comparison (1%-10%)
# -------------------------------------------------------

d_low_avg = [dirty_avg[i] for i in low_idx]
d_low_max = [dirty_max[i] for i in low_idx]
d_low_min = [dirty_min[i] for i in low_idx]
d_low_std = [dirty_std[i] for i in low_idx]

fig, ax = plt.subplots(figsize=(8, 5))

for trial_idx in range(NUM_TRIALS):
    y = [dirty_all_trials[i][trial_idx] for i in low_idx]
    ax.plot(low_x, y, color='tomato', alpha=0.08, linewidth=0.8)

ax.plot(low_x, d_low_avg, color='tomato', linewidth=2.5,
        marker='s', markersize=6, label='Average', zorder=5)

ax.errorbar(low_x, d_low_avg,
            yerr=d_low_std,
            fmt='none', color='tomato', alpha=0.5,
            capsize=4, linewidth=1.2, zorder=4)

annotate_points(ax, low_x, d_low_avg, d_low_max, d_low_min,
                color='tomato', std_vals=d_low_std, fontsize=7, offset=12)

ax.set_xticks(low_x)
ax.set_xticklabels(['1%', '2%', '5%', '10%'])
ax.set_xlabel('Dirty Comparison Error Rate (%)')
ax.set_ylabel('Inversion Ratio (inversions / C(n,2))')
ax.set_title(f'Dirty Comparison: Low Error Rates\n'
             f'(n={n}, {NUM_TRIALS} trials each)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'avg_dirty_1_10.png', dpi=150)
plt.close()
print("Plot 4 saved: avg_dirty_1_10.png")