# Dataset Generation Workflow (2-Phase)

## Overview

**Phase 1:** Bulk processing với `fill_neutrals_local.py`  
**Phase 2:** Cleanup missing rows với `fill_neutrals_cleanup.py`

---

## Phase 1: Bulk Processing (ĐANG CHẠY)

```bash
# Script chính - batch processing
python fill_neutrals_local.py
```

**Đặc điểm:**
- Batch size: 20 rows/batch
- Tốc độ: ~50-60s/batch, ~1 batch/min
- Skip: ~1-2% batches nếu fail (20 rows/batch bị bỏ qua)
- Output: `checkpoint.tsv` (auto-save mỗi batch)

**Để chạy tiếp:** Script sẽ tự resume từ checkpoint nếu dừng giữa chừng.

---

## Phase 2: Cleanup Missing Rows (SAU KHI PHASE 1 XONG)

### Bước 1: Kiểm tra còn bao nhiêu rows thiếu

```bash
python check_missing.py
```

**Output ví dụ:**
```
Total rows: 11927
Missing neutrals: 127 rows
Completion rate: 98.94%

First 20 missing indices: [45, 67, 89, ...]
```

### Bước 2: Chạy cleanup script

```bash
python fill_neutrals_cleanup.py
```

**Đặc điểm:**
- Xử lý **TỪNG ROW riêng lẻ** (không batch)
- Retry: 5 lần/row (vs 3 lần/batch ở Phase 1)
- Adaptive max_tokens: Tự tăng nếu gặp JSON truncation
- Auto-save checkpoint mỗi 10 rows

**Output:**
- Cập nhật trực tiếp `checkpoint.tsv`
- Tạo `cleanup_report.txt` với danh sách rows đã fix

**Ước tính thời gian:**
- ~30-60s/row
- 100 missing rows ≈ 50-100 phút

---

## Workflow Example

```bash
# 1. Chạy phase 1 (đang chạy)
python fill_neutrals_local.py
# → Estimated: 6-7 hours
# → Output: checkpoint.tsv với ~98-99% rows filled

# 2. Sau khi xong, check missing
python check_missing.py
# → Shows: "Missing neutrals: 127 rows"

# 3. Cleanup missing rows
python fill_neutrals_cleanup.py
# → Estimated: ~1-2 hours
# → Output: Updated checkpoint.tsv + cleanup_report.txt

# 4. Verify lần cuối
python check_missing.py
# → Should show: "All rows complete!"

# 5. Final output
cp checkpoint.tsv paradetox_filled.tsv
```

---

## Lợi ích của 2-Phase Approach

### Phase 1 (Batch):
Nhanh, hiệu quả  
Xử lý bulk data tốt  
Throughput cao (~1000 rows/hour)

### Phase 2 (Cleanup):
Không bỏ sót data (0% loss)  
Targeted processing cho missing rows  
Retry tích cực hơn → success rate cao  
Adaptive parameters → xử lý được edge cases

---

## Khi nào cần cleanup?

**Chạy cleanup NẾU:**
- Phase 1 có thông báo `[SKIP] Batch failed after 3 attempts`
- Completion rate < 99% (check với `check_missing.py`)
- Cần dataset hoàn chỉnh 100%

**Không cần NẾU:**
- Completion rate ≥ 99.5%
- Chấp nhận mất ~0.5% data

---

## Troubleshooting

### Cleanup vẫn fail một số rows:
```bash
# Option 1: Chạy lại cleanup (might succeed on retry)
python fill_neutrals_cleanup.py

# Option 2: Tăng max_retries trong script
# Edit fill_neutrals_cleanup.py, line 176:
#   process_single_row(..., max_retries=10)  # Increase from 5 to 10
```

### Muốn process only specific rows:
```python
# Edit fill_neutrals_cleanup.py
# Line 242: fill_indices = [45, 67, 89]  # Manual list
```

---

## Files Generated

```
checkpoint.tsv           # Main output (incremental)
paradetox_filled.tsv     # Final output (copy of checkpoint)
filled_review.csv        # Phase 1 review (newly filled items)
cleanup_report.txt       # Phase 2 report (fixed + failed rows)
```

---

## Tips

1. **Monitor Phase 1:** Xem có nhiều `[SKIP]` không
2. **Check trước Phase 2:** Dùng `check_missing.py` để ước tính thời gian
3. **Patience:** Cleanup chậm hơn nhưng thoroughness cao hơn
4. **Backup:** Trước khi chạy cleanup, backup checkpoint.tsv
   ```bash
   cp checkpoint.tsv checkpoint_backup.tsv
   ```

---

## Summary

| Phase | Speed | Coverage | Use case |
|-------|-------|----------|----------|
| Phase 1 | Fast | ~98-99% | Bulk processing |
| Phase 2 | Slow | 99-100% | Cleanup stragglers |

**Recommended:** Luôn chạy Phase 2 sau Phase 1 để đảm bảo 100% completion.
