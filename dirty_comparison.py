import random
from itertools import combinations

def generate_dirty_comparisons(arr, p):
    """
    Generate dirty comparison predictions for all pairs in arr.
    Each pair (x, y) has its true comparison flipped with probability p.
    為陣列中所有配對生成 dirty comparison 預測。
    每一對 (x, y) 的真實比較結果以機率 p 被翻轉。
    
    Returns / 回傳:
        comparisons: dict mapping (x, y) to predicted result (True means x < y)
                     字典，將每對 (x, y) 對應到預測結果（True 代表 x < y）
        actual_error_rate: fraction of pairs that were flipped
                           實際被翻轉的配對比例
    """
    n = len(arr)
    
    # 計算真實排序 / Compute the true sorted order
    true_order = sorted(arr)
    
    comparisons = {}
    flip_count = 0    # 記錄翻轉次數 / Count how many pairs were flipped
    total_pairs = 0   # 記錄總配對數 / Count total number of pairs

    # 遍歷所有 C(n,2) 個配對 / Iterate over all C(n,2) pairs
    for x, y in combinations(arr, 2):
        
        # 計算真實比較結果：x 在排序後是否排在 y 前面
        # Compute true comparison result: does x come before y in sorted order?
        true_result = (true_order.index(x) < true_order.index(y))
        
        # 以機率 p 翻轉比較結果 / Flip the comparison result with probability p
        if random.random() < p:
            # 翻轉：預測結果與真實結果相反
            # Flip: predicted result is the opposite of the true result
            comparisons[(x, y)] = not true_result
            flip_count += 1
        else:
            # 不翻轉：預測結果與真實結果相同
            # No flip: predicted result matches the true result
            comparisons[(x, y)] = true_result
        
        total_pairs += 1

    # 計算實際錯誤率 = 翻轉數 / 總配對數
    # Compute actual error rate = number of flips / total number of pairs
    actual_error_rate = flip_count / total_pairs
    
    return comparisons, actual_error_rate


# --- 測試 / Test ---
if __name__ == "__main__":
    
    # 固定隨機種子以確保結果可重現
    # Fix random seed to ensure reproducibility
    # 註解掉下一行時，每次執行都會產生不同結果。
    # When the next line is commented out, every run produces different results.
    # random.seed(42)
    
    # 設定陣列大小 / Set array size
    n = 100
    
    # 產生陣列 [1, 2, ..., 100] / Generate array [1, 2, ..., 100]
    arr = list(range(1, n + 1))
    
    # 計算總配對數 C(100, 2) = 4950
    # Compute total number of pairs C(100, 2) = 4950
    total_pairs = n * (n - 1) // 2

    print(f"n = {n}, total pairs = {total_pairs}\n")
    print(f"{'Target p':>12} {'Actual error rate':>20} {'Flipped pairs':>15}")
    print("-" * 50)

    # 對四個目標錯誤率各跑一次
    # Run once for each target error rate
    for p in [0.01, 0.02, 0.05, 0.10]:
        
        # 生成 dirty comparisons 並取得實際錯誤率
        # Generate dirty comparisons and retrieve actual error rate
        _, actual_error_rate = generate_dirty_comparisons(arr, p)
        
        # 計算實際翻轉的配對數
        # Compute the actual number of flipped pairs
        flipped = round(actual_error_rate * total_pairs)
        
        print(f"{p:>12.2f} {actual_error_rate:>20.4f} {flipped:>15}")
