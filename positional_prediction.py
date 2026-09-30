import random
import math
from itertools import combinations

def worst_case(n):
    """
    計算 positional prediction 的 worst-case total displacement。
    Compute the worst-case total displacement for positional prediction.
    
    n 為偶數 / n even: n(3n-2) / 4
    n 為奇數 / n odd:  (n-1)(3n+1) / 4
    """
    if n % 2 == 0:
        return n * (3 * n - 2) // 4
    else:
        return (n - 1) * (3 * n + 1) // 4


def generate_positional_prediction(n, error_rate):
    """
    用正態分佈產生 positional prediction。
    Generate positional prediction using normal distribution.

    sigma 由公式推導，使期望總 displacement 等於目標值：
    sigma is derived so that the expected total displacement equals the target:
        E[total displacement] = n * sigma * sqrt(2/pi) = error_rate * worst_case(n)
    因此 / therefore:
        sigma = error_rate * worst_case(n) / (n * sqrt(2/pi))

    每個元素的位移量從 N(0, sigma^2) 獨立取樣，clamp 到 [1, n]。
    Each element's displacement is sampled independently from N(0, sigma^2),
    clamped to [1, n].

    Parameters / 參數:
        n:          陣列大小 / array size
        error_rate: 目標錯誤率 (0 到 1 之間) / target error rate (between 0 and 1)

    Returns / 回傳:
        p_hat:             預測名次列表 / predicted rank list
        actual_error_rate: 實際錯誤率 / actual error rate
    """
    true_ranks = list(range(1, n + 1))

    # 計算 sigma / Compute sigma
    wc = worst_case(n)
    sigma = (error_rate * wc) / (n * math.sqrt(2 / math.pi))

    p_hat = []
    for i in range(n):
        # 從正態分佈取樣位移量 / Sample displacement from normal distribution
        delta = random.gauss(0, sigma)
        new_rank = true_ranks[i] + round(delta)
        # clamp 到 [1, n] / clamp to [1, n]
        new_rank = max(1, min(n, new_rank))
        p_hat.append(new_rank)

    # 計算實際錯誤率 / Compute actual error rate
    actual_displacement = sum(abs(p_hat[i] - true_ranks[i]) for i in range(n))
    actual_error_rate = actual_displacement / wc

    return p_hat, actual_error_rate


def count_inversions(ranks):
    """
    用 merge sort 計算 inversion 數，時間複雜度 O(n log n)。
    Count inversions using merge sort in O(n log n) time.
    回傳 (sorted_list, inversion_count)。
    Returns (sorted_list, inversion_count).
    """
    if len(ranks) <= 1:
        return ranks, 0
    mid = len(ranks) // 2
    left, lc = count_inversions(ranks[:mid])
    right, rc = count_inversions(ranks[mid:])
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


# --- 測試 / Test ---
if __name__ == "__main__":

    n = 100
    true_ranks = list(range(1, n + 1))

    print(f"n = {n}, worst case = {worst_case(n)}\n")
    print(f"{'Target error%':>15} {'Actual error%':>15} {'Inversions':>12}")
    print("-" * 45)

    for error_rate in [0.01, 0.02, 0.05, 0.10]:
        p_hat, actual_error_rate = generate_positional_prediction(n, error_rate)
        _, inv = count_inversions(p_hat)
        print(f"{error_rate:>15.2%} {actual_error_rate:>15.2%} {inv:>12}")