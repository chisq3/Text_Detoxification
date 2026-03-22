#!/usr/bin/env python3
"""
Cleanup script: Xử lý các rows bị thiếu sau khi chạy fill_neutrals_local.py

Mục đích:
- Tìm rows vẫn còn null/empty trong neutral1/2/3
- Xử lý TỪNG ROW riêng lẻ (không batch) → tránh 1 item lỗi kéo theo cả batch
- Retry tích cực hơn với max_retries=5
- Tự động tăng max_tokens nếu gặp JSON truncation

Cách dùng:
    python fill_neutrals_cleanup.py

Output:
    - Cập nhật trực tiếp vào checkpoint.tsv (hoặc paradetox_filled.tsv)
    - Tạo cleanup_report.txt với danh sách rows đã fix
"""

import os
import sys
import json
import re
import time
import pandas as pd
from vllm import LLM, SamplingParams

# ==============================
# CONFIG
# ==============================

MODEL_PATH      = "Qwen/Qwen2.5-14B-Instruct"
CACHE_DIR       = "/storage/student6/.cache"
INPUT_FILE      = "/storage/student6/NLP/checkpoint.tsv"  # Hoặc paradetox_filled.tsv
OUTPUT_FILE     = INPUT_FILE  # Ghi đè lên file gốc
REPORT_FILE     = "/storage/student6/NLP/cleanup_report.txt"

os.environ["HF_HOME"]               = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"]= "false"

# ==============================
# VALIDATION
# ==============================

def validate_item(item, orig_toxic, orig_n1, orig_n2, orig_n3):
    """Validate 1 item - relaxed validation."""
    if not isinstance(item, dict):
        return False, "Not a dict"
    
    for col in ["neutral1", "neutral2", "neutral3"]:
        val = item.get(col, "")
        if not val or not isinstance(val, str) or len(val.strip()) < 3:
            return False, f"{col} empty or too short"
    
    # Check preserved fields (if not originally null)
    if orig_n1 and item.get("neutral1", "").strip() != orig_n1:
        # Model changed existing field - might be improvement, warn only
        pass
    
    return True, ""

# ==============================
# PROMPT BUILDER
# ==============================

def build_prompt_single(toxic, neutral1, neutral2, neutral3):
    """Build prompt for single item."""
    # Convert empty strings to null for JSON
    n1 = neutral1 if neutral1 else None
    n2 = neutral2 if neutral2 else None
    n3 = neutral3 if neutral3 else None
    
    data = {
        "id": 0,
        "toxic": toxic,
        "neutral1": n1,
        "neutral2": n2,
        "neutral3": n3
    }
    
    data_str = json.dumps([data], indent=2, ensure_ascii=False)
    
    system = "You are a text detoxification expert. Generate neutral paraphrases."
    user = f"""RULES (follow EXACTLY):
1. Output format: Pure JSON array with exactly 1 object.
2. Field "id": Always 0.
3. Fields NOT null in input: Copy EXACTLY as-is.
4. Fields that ARE null: Write NEW detoxified paraphrase.
5. ALL three neutral fields must be non-empty strings.

WRITING GUIDELINES:
- Remove ALL offensive/vulgar/hateful language
- Keep core meaning intact
- Make each paraphrase CLEARLY different
- Natural, grammatically correct English
- Similar length to toxic sentence

INPUT:
{data_str}

OUTPUT (JSON array only):"""
    
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ==============================
# SINGLE ROW PROCESSOR
# ==============================

def process_single_row(llm, row_idx, toxic, n1, n2, n3, max_retries=5, max_tokens=2048):
    """Process 1 row with aggressive retry strategy."""
    
    prompt = build_prompt_single(toxic, n1, n2, n3)
    
    current_max_tokens = max_tokens
    
    for attempt in range(max_retries):
        sampling_params = SamplingParams(
            temperature=0.15,
            top_p=0.9,
            repetition_penalty=1.05,
            max_tokens=current_max_tokens,
        )
        
        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()
        
        # Clean markdown
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list) or len(parsed) == 0:
                print(f"    [Attempt {attempt+1}/{max_retries}] Invalid array")
                continue
            
            item = parsed[0]
            ok, err = validate_item(item, toxic, n1, n2, n3)
            
            if ok:
                return {
                    "neutral1": item["neutral1"],
                    "neutral2": item["neutral2"],
                    "neutral3": item["neutral3"]
                }
            else:
                print(f"    [Attempt {attempt+1}/{max_retries}] Validation: {err}")
                
        except json.JSONDecodeError as e:
            print(f"    [Attempt {attempt+1}/{max_retries}] JSON error: {str(e)[:50]}")
            
            # If truncation suspected, increase max_tokens
            if "Expecting" in str(e) and current_max_tokens < 4096:
                current_max_tokens += 512
                print(f"      → Increasing max_tokens to {current_max_tokens}")
    
    print(f"    [FAILED] Row {row_idx} failed after {max_retries} attempts")
    return None

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    
    print("="*60)
    print("CLEANUP SCRIPT: Fill missing neutrals")
    print("="*60)
    
    # Check file exists
    if not os.path.exists(INPUT_FILE):
        print(f"✗ File not found: {INPUT_FILE}")
        print("  Make sure fill_neutrals_local.py has run first!")
        exit(1)
    
    # Load data
    print(f"\n[1/5] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"  Total rows: {len(df)}")
    
    # Ensure columns exist
    for col in ["neutral1", "neutral2", "neutral3"]:
        if col not in df.columns:
            df[col] = ""
    
    # Find rows needing fill
    def needs_fill(row):
        for col in ["neutral1", "neutral2", "neutral3"]:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                return True
        return False
    
    fill_indices = df[df.apply(needs_fill, axis=1)].index.tolist()
    
    if not fill_indices:
        print("\n✓ Nothing to fill! All rows complete.")
        exit(0)
    
    print(f"\n[2/5] Found {len(fill_indices)} rows needing fill")
    print(f"  Indices: {fill_indices[:10]}{'...' if len(fill_indices) > 10 else ''}")
    
    # Load model
    print(f"\n[3/5] Loading Qwen2.5-14B-Instruct (4-bit)...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="half",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=6144,
        max_num_seqs=16,  # Lower for single-item processing
        disable_log_stats=True,
    )
    print("  Model loaded!")
    
    # Process one by one
    print(f"\n[4/5] Processing {len(fill_indices)} rows (one at a time)...")
    print("  Strategy: Individual retry with max_retries=5, adaptive max_tokens\n")
    
    fixed_rows = []
    failed_rows = []
    start_time = time.time()
    
    for i, idx in enumerate(fill_indices):
        row = df.loc[idx]
        toxic = str(row["toxic"])
        n1 = "" if pd.isna(row["neutral1"]) else str(row["neutral1"]).strip()
        n2 = "" if pd.isna(row["neutral2"]) else str(row["neutral2"]).strip()
        n3 = "" if pd.isna(row["neutral3"]) else str(row["neutral3"]).strip()
        
        print(f"  [{i+1}/{len(fill_indices)}] Row {idx}: {toxic[:50]}...")
        
        result = process_single_row(llm, idx, toxic, n1, n2, n3)
        
        if result:
            # Update df
            if not n1:
                df.at[idx, "neutral1"] = result["neutral1"]
            if not n2:
                df.at[idx, "neutral2"] = result["neutral2"]
            if not n3:
                df.at[idx, "neutral3"] = result["neutral3"]
            
            fixed_rows.append({
                "row_idx": idx,
                "toxic": toxic,
                "neutral1": result["neutral1"],
                "neutral2": result["neutral2"],
                "neutral3": result["neutral3"]
            })
            print(f"    ✓ Fixed")
        else:
            failed_rows.append({"row_idx": idx, "toxic": toxic})
            print(f"    ✗ Failed")
        
        # Save checkpoint every 10 rows
        if (i + 1) % 10 == 0:
            df.to_csv(OUTPUT_FILE, sep="\t", index=False)
            print(f"    [Checkpoint saved: {i+1}/{len(fill_indices)}]")
    
    elapsed = time.time() - start_time
    
    # Final save
    print(f"\n[5/5] Saving final output: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    
    # Generate report
    print(f"  Generating report: {REPORT_FILE}")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CLEANUP REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total rows attempted: {len(fill_indices)}\n")
        f.write(f"Successfully fixed:   {len(fixed_rows)}\n")
        f.write(f"Failed:               {len(failed_rows)}\n")
        f.write(f"Success rate:         {len(fixed_rows)/len(fill_indices)*100:.1f}%\n")
        f.write(f"Time elapsed:         {elapsed/60:.1f} minutes\n\n")
        
        if fixed_rows:
            f.write("="*60 + "\n")
            f.write("FIXED ROWS:\n")
            f.write("="*60 + "\n\n")
            for item in fixed_rows[:50]:  # First 50
                f.write(f"Row {item['row_idx']}:\n")
                f.write(f"  Toxic: {item['toxic']}\n")
                f.write(f"  N1: {item['neutral1']}\n")
                f.write(f"  N2: {item['neutral2']}\n")
                f.write(f"  N3: {item['neutral3']}\n\n")
            
            if len(fixed_rows) > 50:
                f.write(f"... and {len(fixed_rows) - 50} more\n\n")
        
        if failed_rows:
            f.write("="*60 + "\n")
            f.write("FAILED ROWS (still need manual review):\n")
            f.write("="*60 + "\n\n")
            for item in failed_rows:
                f.write(f"Row {item['row_idx']}: {item['toxic']}\n")
    
    # Summary
    print("\n" + "="*60)
    print("CLEANUP COMPLETE!")
    print("="*60)
    print(f"✓ Fixed:   {len(fixed_rows)}/{len(fill_indices)} rows")
    print(f"✗ Failed:  {len(failed_rows)}/{len(fill_indices)} rows")
    print(f"⏱  Time:   {elapsed/60:.1f} minutes")
    print(f"📝 Report: {REPORT_FILE}")
    print("="*60)
    
    if failed_rows:
        print("\n⚠️  Some rows still failed even with aggressive retry.")
        print("   Options:")
        print("   1. Run this script again (might succeed on retry)")
        print("   2. Manually review failed_rows in report")
        print("   3. Increase max_retries in script (currently 5)")
