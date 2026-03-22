#!/usr/bin/env python3
"""
Quick check: Xem có bao nhiêu rows vẫn còn thiếu neutral1/2/3

Cách dùng:
    python check_missing.py [file.tsv]
    
Mặc định: checkpoint.tsv
"""

import sys
import pandas as pd

# File path
if len(sys.argv) > 1:
    FILE = sys.argv[1]
else:
    FILE = "/storage/student6/NLP/checkpoint.tsv"

print(f"Checking: {FILE}\n")

try:
    df = pd.read_csv(FILE, sep="\t")
except Exception as e:
    print(f"  Error loading file: {e}")
    exit(1)

print(f"Total rows: {len(df)}")

# Ensure columns
for col in ["neutral1", "neutral2", "neutral3"]:
    if col not in df.columns:
        df[col] = ""

# Count missing
def needs_fill(row):
    for col in ["neutral1", "neutral2", "neutral3"]:
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            return True
    return False

missing_mask = df.apply(needs_fill, axis=1)
missing_indices = df[missing_mask].index.tolist()

print(f"Missing neutrals: {len(missing_indices)} rows")

if missing_indices:
    print(f"\nCompletion rate: {(len(df) - len(missing_indices)) / len(df) * 100:.2f}%")
    print(f"\nFirst 20 missing indices: {missing_indices[:20]}")
    
    # Show samples
    print("SAMPLE MISSING ROWS:")
    for idx in missing_indices[:5]:
        row = df.loc[idx]
        print(f"\nRow {idx}:")
        print(f"  Toxic: {row['toxic'][:70]}")
        print(f"  N1: {'[MISSING]' if pd.isna(row['neutral1']) or not str(row['neutral1']).strip() else str(row['neutral1'])[:60]}")
        print(f"  N2: {'[MISSING]' if pd.isna(row['neutral2']) or not str(row['neutral2']).strip() else str(row['neutral2'])[:60]}")
        print(f"  N3: {'[MISSING]' if pd.isna(row['neutral3']) or not str(row['neutral3']).strip() else str(row['neutral3'])[:60]}")
    
    print("NEXT STEP:")
    print("  python fill_neutrals_cleanup.py")
else:
    print("\n  All rows complete! No cleanup needed.")
