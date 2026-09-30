#!/opt/homebrew/bin/python3.12
import sys
sys.path.append('/Users/johnsonchang/Desktop/dissertation materials/sorting with predictions')

import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from positional_prediction import generate_positional_prediction, count_inversions, worst_case
from dirty_comparison import generate_dirty_comparisons

# -------------------------------------------------------
# Common settings
# -------------------------------------------------------

# random.seed(42)

n = 1000

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
# Plot 1: Positional Prediction (1%-50%)
# -------------------------------------------------------

MAX_PAIRS = n * (n - 1) // 2
positional_inversions = []

for error_rate in error_rates:
    p_hat, actual_error_rate = generate_positional_prediction_normal(n, error_rate)
    _, inv = merge_count(p_hat)
    ratio = inv / MAX_PAIRS
    positional_inversions.append(ratio)
    print(f"[Positional] target={error_rate:.2%}, actual={actual_error_rate:.2%}, "
          f"inversions={inv}, ratio={ratio:.4f}")

x = [r * 100 for r in error_rates]

fig = plt.figure(figsize=(8, 5))
plt.plot(x, positional_inversions,
         marker='o', color='steelblue', linewidth=2, markersize=6)
for r, ratio in zip(error_rates, positional_inversions):
    plt.annotate(
        f'{ratio*100:.1f}%',
        xy=(r * 100, ratio),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center', fontsize=7, color='steelblue'
    )
plt.xlabel('Target Error Parameter (%)')
plt.ylabel('Inversion Ratio (inversions / C(n,2))')
plt.title(f'Positional Prediction: Error Rate vs Inversion Ratio (n={n})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'single_positional_1_50.png', dpi=150)
plt.close()
print("Plot 1 saved: single_positional_1_50.png\n")

# -------------------------------------------------------
# Plot 2: Positional Prediction (1%-10%)
# -------------------------------------------------------

low_positional = [positional_inversions[error_rates.index(r)] for r in low_error_rates]
low_x = [r * 100 for r in low_error_rates]

fig = plt.figure(figsize=(8, 5))
plt.plot(low_x, low_positional,
         marker='o', color='steelblue', linewidth=2, markersize=7)
for r, ratio in zip(low_error_rates, low_positional):
    plt.annotate(
        f'{ratio*100:.1f}%',
        xy=(r * 100, ratio),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center', fontsize=9, color='steelblue'
    )
plt.xticks(low_x, ['1%', '2%', '5%', '10%'])
plt.xlabel('Target Error Parameter (%)')
plt.ylabel('Inversion Ratio (inversions / C(n,2))')
plt.title(f'Positional Prediction: Low Error Rates (n={n})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'single_positional_1_10.png', dpi=150)
plt.close()
print("Plot 2 saved: single_positional_1_10.png\n")

# -------------------------------------------------------
# Plot 3: Dirty Comparison (1%-50%)
# -------------------------------------------------------

arr = list(range(1, n + 1))
true_order = sorted(arr)
dirty_inversions = []

for error_rate in error_rates:
    comparisons, actual_error_rate = generate_dirty_comparisons(arr, error_rate)
    predicted_order = out_degree_sort(arr, comparisons)
    inv = inversions_from_order(predicted_order, true_order)
    ratio = inv / MAX_PAIRS
    dirty_inversions.append(ratio)
    print(f"[Dirty] target={error_rate:.2%}, actual={actual_error_rate:.2%}, "
          f"inversions={inv}, ratio={ratio:.4f}")

fig = plt.figure(figsize=(8, 5))
plt.plot(x, dirty_inversions,
         marker='s', color='tomato', linewidth=2, markersize=6)
for r, ratio in zip(error_rates, dirty_inversions):
    plt.annotate(
        f'{ratio*100:.1f}%',
        xy=(r * 100, ratio),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center', fontsize=7, color='tomato'
    )
plt.xlabel('Dirty Comparison Error Rate (%)')
plt.ylabel('Inversion Ratio (inversions / C(n,2))')
plt.title(f'Dirty Comparison: Error Rate vs Inversion Ratio (n={n})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'single_dirty_1_50.png', dpi=150)
plt.close()
print("Plot 3 saved: single_dirty_1_50.png\n")

# -------------------------------------------------------
# Plot 4: Dirty Comparison (1%-10%)
# -------------------------------------------------------

low_dirty = [dirty_inversions[error_rates.index(r)] for r in low_error_rates]

fig = plt.figure(figsize=(8, 5))
plt.plot(low_x, low_dirty,
         marker='s', color='tomato', linewidth=2, markersize=7)
for r, ratio in zip(low_error_rates, low_dirty):
    plt.annotate(
        f'{ratio*100:.1f}%',
        xy=(r * 100, ratio),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center', fontsize=9, color='tomato'
    )
plt.xticks(low_x, ['1%', '2%', '5%', '10%'])
plt.xlabel('Dirty Comparison Error Rate (%)')
plt.ylabel('Inversion Ratio (inversions / C(n,2))')
plt.title(f'Dirty Comparison: Low Error Rates (n={n})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'single_dirty_1_10.png', dpi=150)
plt.close()
print("Plot 4 saved: single_dirty_1_10.png")