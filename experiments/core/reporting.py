import os
from typing import Callable, Dict, List

import pandas as pd


TextWriter = Callable[[List[Dict[str, float]]], str]


def export_leaderboard(
    results: List[Dict[str, float]],
    output_dir: str,
    csv_name: str,
    json_name: str,
    txt_name: str,
    text_writer: TextWriter,
) -> None:
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, csv_name)
    json_path = os.path.join(output_dir, json_name)
    txt_path = os.path.join(output_dir, txt_name)

    frame = pd.DataFrame(results)
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_writer(results))

    print(f"\nLeaderboard exported to: {txt_path}")
    print(f"Leaderboard exported to: {csv_path}")
    print(f"Leaderboard exported to: {json_path}")
