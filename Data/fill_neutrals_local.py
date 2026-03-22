"""
fill_neutrals_local.py
======================
Qwen2.5-14B-Instruct local, 1 GPU RTX 3090.
- Single GPU → tránh hoàn toàn NCCL/multi-process segfault
- 4-bit quantization (AWQ) → ~8GB VRAM, fit thoải mái trong 24GB
- Cache redirect sang /storage tránh /home hết dung lượng
"""

import json
import os
import re
import time
import pandas as pd

# REDIRECT CACHE TRƯỚC KHI IMPORT vllm
# PHẢI set env vars trước khi import bất kỳ lib nào
CACHE_DIR = "/storage/student6/.cache"
for d in ["vllm", "flashinfer", "huggingface", "triton"]:
    os.makedirs(f"{CACHE_DIR}/{d}", exist_ok=True)

os.environ["HOME"]                     = "/storage/student6"
os.environ["VLLM_CACHE_ROOT"]          = f"{CACHE_DIR}/vllm"
os.environ["FLASHINFER_WORKSPACE_DIR"] = f"{CACHE_DIR}/flashinfer"
os.environ["HF_HOME"]                  = f"{CACHE_DIR}/huggingface"
os.environ["TRANSFORMERS_CACHE"]       = f"{CACHE_DIR}/huggingface"
os.environ["TRITON_CACHE_DIR"]         = f"{CACHE_DIR}/triton"
os.environ["XDG_CACHE_HOME"]           = CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"]     = "1"      # 1 GPU duy nhất → không cần NCCL
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["VLLM_LOGGING_LEVEL"]       = "WARNING"

from vllm import LLM, SamplingParams

# CONFIG

# Chọn model phù hợp VRAM:
#   bfloat16 full  → ~28GB → KHÔNG vừa 1x RTX3090 (24GB)
#   4-bit AWQ      → ~8GB  → vừa thoải mái ✓
#
# Nếu đã download full model (bfloat16) mà chưa có AWQ,
# set USE_QUANTIZATION = False và model sẽ tự quantize lúc load
# (chậm hơn một chút nhưng không cần download lại)

MODEL_PATH  = "./models/Qwen2.5-14B-Instruct"
INPUT_FILE  = "paradetox.tsv"
OUTPUT_FILE = "paradetox_filled.tsv"
CHECKPOINT  = "checkpoint.tsv"
BATCH_SIZE  = 20    # giảm batch size → ít validation errors hơn

# VALIDATION

def validate_results(results, expected_count, original_batch):
    """
    Validate LLM output:
    1. Correct structure (list of dicts with required keys)
    2. No empty/null values in neutral fields
    3. For non-null input fields: accept model improvements (spelling, completion, normalization)
    """
    if not isinstance(results, list):
        return False, f"Not a list, got {type(results)}"
    if len(results) != expected_count:
        return False, f"Expected {expected_count} items, got {len(results)}"
    
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            return False, f"Item {i} is not a dict"
        
        # Check required keys
        for key in ["id", "neutral1", "neutral2", "neutral3"]:
            if key not in item:
                return False, f"Item {i} missing key '{key}'"
        
        # Check id matches
        if item["id"] != original_batch[i]["id"]:
            return False, f"Item {i} id mismatch: expected {original_batch[i]['id']}, got {item['id']}"
        
        # Check all neutral fields are non-empty strings
        for col in ["neutral1", "neutral2", "neutral3"]:
            val = item[col]
            if not isinstance(val, str) or not val.strip():
                return False, f"Item {i} field '{col}' empty/null: {repr(val)}"
        
        # RELAXED: Don't validate exact copy for non-null fields
        # Reason: Model often fixes spacing, completes incomplete sentences, fixes spelling
        # These improvements make the dataset better quality
        # We only validate that:
        # 1. Something is there (checked above)
        # 2. Toxic content is not added back (semantic check would be complex, skip for now)
    
    return True, ""

# PROMPT BUILDER
def build_prompt(batch):
    items = []
    for item in batch:
        items.append({
            "id":       item["id"],
            "toxic":    item["toxic"],
            "neutral1": item["neutral1"] if item["neutral1"] else None,
            "neutral2": item["neutral2"] if item["neutral2"] else None,
            "neutral3": item["neutral3"] if item["neutral3"] else None,
        })

    data_str = json.dumps(items, ensure_ascii=False)

    system = (
        "You are a text detoxification expert. "
        "You output ONLY valid JSON arrays. No markdown, no explanations, no extra text."
    )

    user = f"""TASK: Fill missing neutral paraphrases in the JSON data below.

INPUT FORMAT: Array of {len(batch)} items. Each has:
- "toxic": offensive sentence
- "neutral1", "neutral2", "neutral3": detoxified versions (some are null)

OUTPUT FORMAT: JSON array with {len(batch)} objects, each with exactly 4 keys: "id", "neutral1", "neutral2", "neutral3"

RULES (follow EXACTLY):
1. Output format: Pure JSON array only. No ```json```, no explanations, nothing else.
2. Array length: Exactly {len(batch)} objects, same order as input.
3. Field "id": Copy from input unchanged (must be integer).
4. Fields NOT null in input: Copy to output EXACTLY as-is (same capitalization, punctuation, spaces).
5. Fields that ARE null: Write a new detoxified paraphrase (see writing guidelines below).
6. ALL three neutral fields in output must be non-empty strings. Never null, never "".

WRITING GUIDELINES (for null fields only):
- Remove ALL offensive/vulgar/hateful/aggressive language from the toxic sentence
- Keep the core meaning and intent intact
- Make each paraphrase clearly different from the others in the same item
- Match the tone/formality of existing neutral fields when available
- Use natural, grammatically correct English
- No added opinions or information beyond what's in the toxic sentence
- Similar length to the toxic sentence

EXAMPLES:

Example 1 - All fields null:
Input:  [{{"id":0,"toxic":"shut the fuck up","neutral1":null,"neutral2":null,"neutral3":null}}]
Output: [{{"id":0,"neutral1":"please be quiet","neutral2":"could you stop talking","neutral3":"I would appreciate some silence"}}]

Example 2 - One field filled (MOST COMMON - pay attention):
Input:  [{{"id":0,"toxic":"you are such an idiot","neutral1":"you are mistaken","neutral2":null,"neutral3":null}}]
Output: [{{"id":0,"neutral1":"you are mistaken","neutral2":"you might be wrong about this","neutral3":"I think you're not understanding correctly"}}]

Example 3 - Two fields filled:
Input:  [{{"id":0,"toxic":"this is stupid garbage","neutral1":"this is not good","neutral2":"this is unsuitable","neutral3":null}}]
Output: [{{"id":0,"neutral1":"this is not good","neutral2":"this is unsuitable","neutral3":"this is not appropriate"}}]

NOW PROCESS THIS INPUT:
{data_str}

OUTPUT (JSON array only):"""

    # Qwen2.5 chat template
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# INFERENCE

def run_batch(batch, llm, sampling_params, max_retries=3):
    prompt = build_prompt(batch)

    for attempt in range(max_retries):
        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$",           "", text).strip()

        try:
            parsed = json.loads(text)
            ok, err = validate_results(parsed, len(batch), batch)
            if ok:
                return parsed
            print(f"\n  [Validation fail {attempt+1}/{max_retries}] {err}")
        except json.JSONDecodeError as e:
            print(f"\n  [JSON error {attempt+1}/{max_retries}] {e}")
            print(f"  Raw (300 chars): {text[:300]}")

    print(f"\n  [SKIP] Batch failed after {max_retries} attempts")
    return None

# MAIN — bắt buộc có guard này để vllm spawn worker đúng cách

if __name__ == "__main__":

    # KIỂM TRA bitsandbytes
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes {bitsandbytes.__version__} sẵn sàng")
    except ImportError:
        print("✗ Thiếu bitsandbytes! Chạy lệnh sau rồi thử lại:")
        print("  pip install bitsandbytes")
        exit(1)

    # LOAD MODEL (4-bit bitsandbytes)

    print(f"Loading Qwen2.5-14B-Instruct 4-bit trên GPU 1...")
    print(f"VRAM cần: ~8GB / 24GB available")
    print(f"Cache dir: {CACHE_DIR}\n")

    llm = LLM(
        model=MODEL_PATH,
        dtype="half",                  # float16 base, weights nén 4-bit
        quantization="bitsandbytes",   # 4-bit NF4 quantization
        load_format="bitsandbytes",
        tensor_parallel_size=1,        # single GPU → không NCCL, không segfault
        gpu_memory_utilization=0.85,   # 85% × 24GB = 20.4GB — dư cho KV cache
        enforce_eager=True,            # không cần flash-attn
        enable_prefix_caching=False,
        max_model_len=6144,
        max_num_seqs=32,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.15,          # balanced: diverse nhưng vẫn consistent
        top_p=0.9,
        repetition_penalty=1.05,
        max_tokens=2048,           # đủ cho batch 20
    )

    print("Model loaded!\n")

    # Load data (resume từ checkpoint nếu có)
    if os.path.exists(CHECKPOINT):
        print(f"[RESUME] Loading checkpoint: {CHECKPOINT}")
        df = pd.read_csv(CHECKPOINT, sep="\t")
    else:
        print(f"[START] Loading: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE, sep="\t")

    print(f"Total rows: {len(df)}")

    for col in ["neutral1", "neutral2", "neutral3"]:
        if col not in df.columns:
            df[col] = ""

    # Tìm rows cần fill
    def needs_fill(row):
        for col in ["neutral1", "neutral2", "neutral3"]:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                return True
        return False

    fill_indices = df[df.apply(needs_fill, axis=1)].index.tolist()
    print(f"Rows needing fill: {len(fill_indices)}")

    if not fill_indices:
        print("Nothing to fill! Saving...")
        df.to_csv(OUTPUT_FILE, sep="\t", index=False)
        exit(0)

    # Main loop
    filled        = 0
    errors        = 0
    filled_items  = []  # Track filled items for review
    total_batches = (len(fill_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    CHECKPOINT_TMP = CHECKPOINT + ".tmp"

    def save_checkpoint(df, path, tmp_path):
        """Atomic save: ghi .tmp trước, rename sau → không bao giờ corrupt."""
        df.to_csv(tmp_path, sep="\t", index=False)
        os.replace(tmp_path, path)

    print(f"Starting: {total_batches} batches × {BATCH_SIZE} rows each")
    print(f"Estimated time: ~{total_batches * 0.5:.1f} minutes (30s/batch average)\n")

    start_time = time.time()
    for batch_num, batch_start in enumerate(range(0, len(fill_indices), BATCH_SIZE)):
        batch_idx = fill_indices[batch_start : batch_start + BATCH_SIZE]

        batch = []
        for i, idx in enumerate(batch_idx):
            row = df.loc[idx]
            batch.append({
                "id":       i,
                "toxic":    str(row["toxic"]),
                "neutral1": "" if pd.isna(row["neutral1"]) else str(row["neutral1"]).strip(),
                "neutral2": "" if pd.isna(row["neutral2"]) else str(row["neutral2"]).strip(),
                "neutral3": "" if pd.isna(row["neutral3"]) else str(row["neutral3"]).strip(),
            })
        
        # Preview first batch for verification
        if batch_num == 0:
            print("=" * 60)
            print("FIRST BATCH SAMPLE (verify logic):")
            print("=" * 60)
            for j in range(min(2, len(batch))):
                print(f"Item {j}:")
                print(f"  Toxic: {batch[j]['toxic'][:60]}...")
                print(f"  N1: {'[FILL]' if not batch[j]['neutral1'] else batch[j]['neutral1'][:60]}")
                print(f"  N2: {'[FILL]' if not batch[j]['neutral2'] else batch[j]['neutral2'][:60]}")
                print(f"  N3: {'[FILL]' if not batch[j]['neutral3'] else batch[j]['neutral3'][:60]}")
            print("=" * 60 + "\n")

        batch_start_time = time.time()
        results = run_batch(batch, llm, sampling_params)

        if results is None:
            errors += 1
        else:
            # Preview output của batch đầu để verify quality
            if batch_num == 0:
                print("FIRST BATCH OUTPUT (check diversity & quality):")
                for j in range(min(2, len(results))):
                    print(f"\nItem {j}:")
                    print(f"  Toxic: {batch[j]['toxic'][:70]}")
                    print(f"  N1: {results[j]['neutral1'][:70]}")
                    print(f"  N2: {results[j]['neutral2'][:70]}")
                    print(f"  N3: {results[j]['neutral3'][:70]}")
                print("   Press Ctrl+C now if output looks bad, or wait 5s to continue...")
                try:
                    time.sleep(5)
                except KeyboardInterrupt:
                    print("\n\n⚠️  Stopped by user. Checkpoint saved, you can resume later.")
                    exit(0)
                print("   Continuing...\n")
            
            for item in results:
                local_id = item.get("id")
                if local_id is None or local_id >= len(batch_idx):
                    continue
                orig_idx = batch_idx[local_id]
                orig_row = batch[local_id]
                
                # Track what was filled for review
                filled_in_item = {
                    "row_idx": orig_idx,
                    "toxic": orig_row["toxic"],
                    "neutral1_old": orig_row["neutral1"],
                    "neutral1_new": item.get("neutral1", ""),
                    "neutral2_old": orig_row["neutral2"],
                    "neutral2_new": item.get("neutral2", ""),
                    "neutral3_old": orig_row["neutral3"],
                    "neutral3_new": item.get("neutral3", ""),
                    "filled_fields": []
                }
                
                for col in ["neutral1", "neutral2", "neutral3"]:
                    new_val = item.get(col, "")
                    if new_val and not orig_row[col]:
                        df.at[orig_idx, col] = new_val
                        filled += 1
                        filled_in_item["filled_fields"].append(col)
                
                # Only save if something was filled
                if filled_in_item["filled_fields"]:
                    filled_items.append(filled_in_item)

            # Lưu checkpoint ngay sau mỗi batch thành công
            save_checkpoint(df, CHECKPOINT, CHECKPOINT_TMP)

        done = batch_num + 1
        pct  = done / total_batches * 100
        elapsed = time.time() - start_time
        speed = done / elapsed * 60 if elapsed > 0 else 0
        eta_min = (total_batches - done) / (done / elapsed) / 60 if elapsed > 0 and done > 0 else 0
        
        print(f"  [{done}/{total_batches}] {pct:.1f}% | Filled: {filled} | Errors: {errors} | "
              f"Speed: {speed:.1f} batch/min | ETA: {eta_min:.1f}min", end="\r")

    # Final save
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    total_time = time.time() - start_time
    
    # Save filled items for quick review
    if filled_items:
        review_data = []
        for item in filled_items:
            row = {
                "row_idx": item["row_idx"],
                "toxic": item["toxic"],
                "filled_fields": ",".join(item["filled_fields"])
            }
            # Add filled values
            for col in ["neutral1", "neutral2", "neutral3"]:
                old_key = f"{col}_old"
                new_key = f"{col}_new"
                if col in item["filled_fields"]:
                    row[col] = f"[NEW] {item[new_key]}"
                elif item[old_key]:
                    row[col] = f"[KEPT] {item[old_key]}"
                else:
                    row[col] = ""
            review_data.append(row)
        
        review_df = pd.DataFrame(review_data)
        review_file = "filled_review.csv"
        review_df.to_csv(review_file, index=False)
        print(f"\n  Review file saved: {review_file} ({len(review_data)} items with new content)")
    
    print(f"  COMPLETED!")
    print(f"Total time      : {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"Cells filled    : {filled}")
    print(f"Batch errors    : {errors} / {total_batches} ({errors/total_batches*100:.1f}%)")
    print(f"Success rate    : {(total_batches-errors)/total_batches*100:.1f}%")
    print(f"Avg time/batch  : {total_time/total_batches:.1f}s")
    print(f"Output file     : {OUTPUT_FILE}")
    print(f"Checkpoint      : {CHECKPOINT}")

    df2  = pd.read_csv(OUTPUT_FILE, sep="\t")
    missing = df2[["neutral1", "neutral2", "neutral3"]].isna().sum().sum()
    total_cells = len(df2) * 3
    completion = (total_cells - missing) / total_cells * 100
    
    print(f"\nDATASET STATS:")
    print(f"  Total rows        : {len(df2)}")
    print(f"  Total cells       : {total_cells}")
    print(f"  Filled cells      : {total_cells - missing}")
    print(f"  Missing cells     : {missing}")
    print(f"  Completion rate   : {completion:.2f}%")
    
    if missing > 0:
        print(f"\n   Still have {missing} empty cells. Run again to retry failed batches.")