"""ラベル済み detections.csv を集計して TP/FP の傾向を分析。

使い方:
    uv run python examples/summarize_detections.py captures/detections.csv

CSV の `label` 列に以下のいずれかを入れてから実行:
    TP        : True Positive  (実際に対象フレーズを発話して検出)
    FP        : False Positive (発話していないのに誤検出)
    uncertain : 判定不能
    (空)      : 未ラベル

`notes` 列に「実際に聞こえた言葉」を書いておくと誤検出の傾向分析に有用。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path, help="detections.csv のパス")
    p.add_argument("--bin-width", type=float, default=0.1,
                   help="スコアヒストグラムのビン幅 (デフォルト: 0.1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: {args.csv_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, str]] = []
    with open(args.csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("レコードがありません")
        return

    n_total = len(rows)
    by_label = Counter(r.get("label", "").strip() or "(未ラベル)" for r in rows)
    by_phrase = Counter(r.get("phrase", "") for r in rows)

    print(f"=== Detection Summary ({args.csv_path}) ===")
    print(f"総検出数: {n_total}")
    print()

    print("[フレーズ別]")
    for phrase, n in by_phrase.most_common():
        print(f"  {phrase:>15}: {n}")
    print()

    print("[ラベル別]")
    for label, n in by_label.most_common():
        print(f"  {label:>12}: {n}  ({100 * n / n_total:.1f}%)")
    print()

    # フレーズ × ラベル
    phrase_label = defaultdict(Counter)
    for r in rows:
        phrase = r.get("phrase", "")
        label = r.get("label", "").strip() or "(未ラベル)"
        phrase_label[phrase][label] += 1

    print("[フレーズ × ラベル]")
    for phrase, counts in phrase_label.items():
        tp = counts.get("TP", 0)
        fp = counts.get("FP", 0)
        unc = counts.get("uncertain", 0)
        unl = counts.get("(未ラベル)", 0)
        labeled = tp + fp + unc
        precision = tp / labeled if labeled > 0 else 0.0
        print(f"  「{phrase}」: TP={tp}  FP={fp}  uncertain={unc}  unlabeled={unl}"
              f"  → precision={precision:.1%} (uncertain除く)")
    print()

    # スコア帯別の TP/FP
    print(f"[スコア帯別 TP/FP] (bin width = {args.bin_width})")
    score_bins: dict[float, Counter] = defaultdict(Counter)
    for r in rows:
        try:
            score = float(r.get("score", "0"))
        except ValueError:
            continue
        bin_lo = (int(score / args.bin_width)) * args.bin_width
        label = r.get("label", "").strip() or "(未ラベル)"
        score_bins[bin_lo][label] += 1

    print(f"  {'score':>10}  {'TP':>5}  {'FP':>5}  {'unc':>5}  {'(未)':>5}  precision")
    for bin_lo in sorted(score_bins.keys()):
        counts = score_bins[bin_lo]
        tp = counts.get("TP", 0)
        fp = counts.get("FP", 0)
        unc = counts.get("uncertain", 0)
        unl = counts.get("(未ラベル)", 0)
        labeled = tp + fp
        precision = tp / labeled if labeled > 0 else 0.0
        rng = f"{bin_lo:.2f}-{bin_lo + args.bin_width:.2f}"
        print(f"  {rng:>10}  {tp:>5}  {fp:>5}  {unc:>5}  {unl:>5}  "
              f"{precision:.1%}" if labeled > 0 else
              f"  {rng:>10}  {tp:>5}  {fp:>5}  {unc:>5}  {unl:>5}  —")
    print()

    # 誤検出時の "実際に聞こえた言葉" 集計
    fp_notes = [r.get("notes", "").strip() for r in rows
                if r.get("label", "").strip() == "FP" and r.get("notes", "").strip()]
    if fp_notes:
        print("[FP の notes 上位 (実際の発話内容)]")
        for note, n in Counter(fp_notes).most_common(20):
            print(f"  {n:>3}回: {note}")
    else:
        print("[FP の notes] 入力なし")


if __name__ == "__main__":
    main()
