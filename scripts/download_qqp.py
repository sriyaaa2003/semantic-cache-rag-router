"""Download the Quora Question Pairs validation split (GLUE) as JSONL for `ragcache eval-threshold`.

Requires pyarrow (`pip install pyarrow`). Source: nyu-mll/glue on the Hugging Face hub.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/nyu-mll/glue/resolve/"
    "refs%2Fconvert%2Fparquet/qqp/validation/0000.parquet"
)


def main(out: str = "data/qqp_validation.jsonl") -> None:
    import pyarrow.parquet as pq

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out).with_suffix(".parquet")
    urllib.request.urlretrieve(URL, tmp)  # noqa: S310 (fixed https URL)
    table = pq.read_table(tmp)
    with open(out, "w", encoding="utf-8") as f:
        for row in table.to_pylist():
            f.write(json.dumps(
                {"q1": row["question1"], "q2": row["question2"], "is_duplicate": bool(row["label"])}
            ) + "\n")
    tmp.unlink()
    print(f"wrote {table.num_rows} pairs to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
