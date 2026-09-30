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

output_dir = Path(__file__).resolve().parent

error_rates = [
    0.01, 0.02, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
]

# n, num_trials, avg_color, trial_color, label
n_configs = [
    (100,  50, 'black',      'grey',       'n=100'),
    (300,  30, 'steelblue',  'lightblue',  'n=300'),
    (600,  20, 'crimson',    'lightpink',  'n=600'),
    (1000, 20, 'darkgreen',  'lightgreen', 'n=1000'),
]

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
# Run experiments for all n configs
# -------------------------------------------------------

pos_results   = {}
dirty_results = {}

x = [r * 100 for r in error_rates]

for n, num_trials, avg_color, trial_color, label in n_configs:
    MAX_PAIRS = n * (n - 1) // 2
    arr        = list(range(1, n + 1))
    true_order = sorted(arr)

    # all_trials[er_idx] = list of num_trials inversion ratios
    pos_all_trials   = [[] for _ in error_rates]
    dirty_all_trials = [[] for _ in error_rates]

    print(f"\n--- n={n}, trials={num_trials} ---")

    for er_idx, error_rate in enumerate(error_rates):
        for _ in range(num_trials):
            # Positional
            p_hat, _ = generate_positional_prediction_normal(n, error_rate)
            _, inv = merge_count(p_hat)
            pos_all_trials[er_idx].append(inv / MAX_PAIRS)

            # Dirty
            comparisons, _ = generate_dirty_comparisons(arr, error_rate)
            predicted_order = out_degree_sort(arr, comparisons)
            true_rank = {v: i for i, v in enumerate(true_order)}
            rank_list = [true_rank[x_] for x_ in predicted_order]
            _, inv = merge_count(rank_list)
            dirty_all_trials[er_idx].append(inv / MAX_PAIRS)

        pos_avg   = sum(pos_all_trials[er_idx]) / num_trials
        dirty_avg = sum(dirty_all_trials[er_idx]) / num_trials
        print(f"  target={error_rate:.2%}  "
              f"pos_inv={pos_avg*100:.2f}%  "
              f"dirty_inv={dirty_avg*100:.2f}%")

    pos_results[n] = {
        'all_trials': pos_all_trials,
        'avg': [sum(t) / num_trials for t in pos_all_trials],
    }
    dirty_results[n] = {
        'all_trials': dirty_all_trials,
        'avg': [sum(t) / num_trials for t in dirty_all_trials],
    }

# -------------------------------------------------------
# Plot 1: Positional — Inversion Ratio vs Error Rate
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for n, num_trials, avg_color, trial_color, label in n_configs:
    all_trials = pos_results[n]['all_trials']
    avg        = pos_results[n]['avg']

    # 畫每次 trial 的淡色折線
    for trial_idx in range(num_trials):
        y = [all_trials[er_idx][trial_idx] * 100
             for er_idx in range(len(error_rates))]
        ax.plot(x, y, color=trial_color, alpha=0.15, linewidth=0.6)

    # 畫平均折線
    ax.plot(x, [r * 100 for r in avg],
            color=avg_color, linewidth=2.2, marker='o', markersize=4,
            label=f'{label} ({num_trials} trials)', zorder=5)

ax.set_xlabel('Target Error Parameter (%)')
ax.set_ylabel('Inversion Ratio (%)')
ax.set_title('Positional Prediction: Inversion Ratio vs Error Rate\n(multiple n)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'scale_positional_inv.png', dpi=150)
plt.close()
print("\nPlot 1 saved: scale_positional_inv.png")

# -------------------------------------------------------
# Plot 2: Dirty — Inversion Ratio vs Error Rate
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for n, num_trials, avg_color, trial_color, label in n_configs:
    all_trials = dirty_results[n]['all_trials']
    avg        = dirty_results[n]['avg']

    for trial_idx in range(num_trials):
        y = [all_trials[er_idx][trial_idx] * 100
             for er_idx in range(len(error_rates))]
        ax.plot(x, y, color=trial_color, alpha=0.15, linewidth=0.6)

    ax.plot(x, [r * 100 for r in avg],
            color=avg_color, linewidth=2.2, marker='s', markersize=4,
            label=f'{label} ({num_trials} trials)', zorder=5)

ax.set_xlabel('Dirty Comparison Error Rate (%)')
ax.set_ylabel('Inversion Ratio (%)')
ax.set_title('Dirty Comparison: Inversion Ratio vs Error Rate\n(multiple n)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'scale_dirty_inv.png', dpi=150)
plt.close()
print("Plot 2 saved: scale_dirty_inv.png")