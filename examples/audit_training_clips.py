"""学習用合成音声の人手チェック支援ツール。

output/<model_name>/{positive,negative}_{train,test}/ から
ランダムにN個ずつ抽出して audit/ にコピーし、各クリップに対して
学習済みモデルの推論スコアもCSVに記録する。

これにより以下が分析できる:
  - 正例 (positive_*) の発音/品質が妥当か → 耳で確認
  - 正例なのにモデルが低スコア → 学習データの問題 or モデル容量不足
  - 負例 (negative_*) なのにモデルが高スコア → adversarial が target に近すぎ
  - 同一フレーズなのにスコアばらつき大 → 合成音声の品質ばらつき

使い方:
    uv run python examples/audit_training_clips.py tomatte_kudasai
    uv run python examples/audit_training_clips.py tomatte_kudasai --n 30
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from livekit.wakeword import WakeWordModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model_name", help="例: tomatte_kudasai")
    p.add_argument("--n", type=int, default=20, help="各 split から抽出する数 (デフォルト: 20)")
    p.add_argument("--root", type=Path, default=Path("output"),
                   help="モデル出力ディレクトリ (デフォルト: output)")
    p.add_argument("--audit-dir", type=Path, default=Path("audit"),
                   help="audit 出力先 (デフォルト: audit)")
    p.add_argument("--seed", type=int, default=42, help="再現可能なサンプリング")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model_dir = args.root / args.model_name
    onnx_path = model_dir / f"{args.model_name}.onnx"

    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} が見つかりません", file=sys.stderr)
        sys.exit(1)
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} が見つかりません (先に export を実行)", file=sys.stderr)
        sys.exit(1)

    model = WakeWordModel(models=[onnx_path])
    stem = onnx_path.stem

    splits = ["positive_train", "positive_test", "negative_train", "negative_test"]
    audit_out = args.audit_dir / args.model_name
    audit_out.mkdir(parents=True, exist_ok=True)

    csv_path = audit_out / "audit.csv"
    rows: list[list[str]] = []
    rows.append(["split", "src_filename", "audit_filename", "score", "audio_seconds",
                 "peak_amplitude", "notes_pronunciation", "notes_quality"])

    print(f"=== Training Clip Audit: {args.model_name} ===")
    print(f"モデル: {onnx_path}")
    print(f"audit 出力先: {audit_out.resolve()}")
    print()

    for split in splits:
        split_dir = model_dir / split
        if not split_dir.is_dir():
            print(f"  [skip] {split}: ディレクトリなし")
            continue

        clips = sorted(p for p in split_dir.iterdir() if p.suffix == ".wav"
                       and p.name.startswith("clip_") and "_r" not in p.name)
        if not clips:
            print(f"  [skip] {split}: クリップなし")
            continue

        sample = random.sample(clips, min(args.n, len(clips)))
        sample.sort()  # 抽出後はファイル名順で見やすく

        scores: list[float] = []
        for src in sample:
            audio, sr = sf.read(str(src), dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]

            duration = len(audio) / sr
            peak = float(np.max(np.abs(audio))) / 32768.0

            chunk = audio
            if sr != 16000:
                # 念のため (通常16k固定)
                import librosa  # type: ignore[import-untyped]
                chunk = librosa.resample(
                    audio.astype(np.float32) / 32768.0,
                    orig_sr=sr, target_sr=16000,
                )
                chunk = (chunk * 32768.0).astype(np.int16)

            pred = model.predict(chunk).get(stem, 0.0)
            scores.append(pred)

            # audit/ にコピー (スコアと出所が分かる名前で)
            dest_name = f"{split}__{src.stem}__score{pred:.3f}.wav"
            dest = audit_out / dest_name
            shutil.copy2(src, dest)
            rows.append([split, src.name, dest_name, f"{pred:.4f}",
                         f"{duration:.2f}", f"{peak:.3f}", "", ""])

        scores_arr = np.array(scores) if scores else np.array([0.0])
        print(f"  [{split}] n={len(sample):>3}  "
              f"score: mean={scores_arr.mean():.3f}  min={scores_arr.min():.3f}  "
              f"max={scores_arr.max():.3f}  median={np.median(scores_arr):.3f}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    print()
    print(f"CSV: {csv_path}")
    print()
    print("【聴取ガイド】audit/ のWAVを開いて以下を確認:")
    print("  - positive_*: 「対象フレーズ」が正しく発音されているか")
    print("  - negative_*: target に近すぎる発音になっていないか")
    print("  - score (ファイル名) と聴感がどの程度合致するか")
    print("  - 同じフレーズなのに声・速度・抑揚が多様か")
    print("  - notes_pronunciation / notes_quality 列に気付きを書き込み")


if __name__ == "__main__":
    main()
