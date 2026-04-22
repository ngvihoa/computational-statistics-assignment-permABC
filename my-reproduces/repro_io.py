"""Xuất kết quả chạy mô phỏng ra file CSV và JSON để dễ dàng phân tích."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from repro_metrics import RunResult


def save_outputs(results: List[RunResult], summary: Dict, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "seed",
                "runtime_sec",
                "eps_final",
                "n_sim_total",
                "mae_global",
                "mae_local_mean",
                "true_local_mean",
                "est_local_mean",
                "true_global",
                "est_global",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.method,
                    row.seed,
                    f"{row.runtime_sec:.6f}",
                    f"{row.eps_final:.6f}",
                    f"{row.n_sim_total:.2f}",
                    f"{row.mae_global:.6f}",
                    f"{row.mae_local_mean:.6f}",
                    f"{row.true_local_mean:.6f}",
                    f"{row.est_local_mean:.6f}",
                    json.dumps(row.true_global.tolist()),
                    json.dumps(row.est_global.tolist()),
                ]
            )

    json_path = out_dir / f"{prefix}_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary}, file, indent=2)
