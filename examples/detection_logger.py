"""検出ロガー — マイク入力をリアルタイム監視し、検出時に音声+CSVを保存。

低めの閾値（例 --threshold 0.05）で borderline 含めて捕まえ、
後で人手 or 音声認識エンジンで TP/FP ラベルを付けて分析することを想定。

出力構成:
    captures/
    ├── detections.csv      ← id, datetime, phrase, score, wav, label(空), notes(空)
    ├── score_history.csv   ← --score-log 指定時のみ。閾値未満も含む全推論ログ
    └── 20260522_HHMMSS_tomatte_kudasai_0.312.wav

使い方:
    uv run python examples/detection_logger.py --threshold 0.05
    uv run python examples/detection_logger.py --threshold 0.05 --pre 2 --post 3 --score-log
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from livekit.wakeword import WakeWordModel

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms / frame (matches WakeWordListener)
CHUNK_SECONDS = 2.0
CHUNK_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)

# モデル登録（フレーズ → ONNXパス）
MODELS = {
    "止まって下さい": Path("output/tomatte_kudasai/tomatte_kudasai.onnx"),
    "こっちに来て":   Path("output/kocchi_ni_kite/kocchi_ni_kite.onnx"),
}

# score_history.csv に書き出す最低スコア
SCORE_LOG_MIN = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="日本語コマンド検出ロガー")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="この値以上のスコアで音声を保存 (デフォルト: 0.05)")
    p.add_argument("--pre", type=float, default=2.0,
                   help="検出ウィンドウより前に保存する秒数 (デフォルト: 2.0)")
    p.add_argument("--post", type=float, default=3.0,
                   help="検出ウィンドウより後に保存する秒数 (デフォルト: 3.0)")
    p.add_argument("--debounce", type=float, default=1.0,
                   help="連続検出を抑制する秒数 (デフォルト: 1.0)")
    p.add_argument("--out", type=Path, default=Path("captures"),
                   help="出力ディレクトリ (デフォルト: captures)")
    p.add_argument("--score-log", action="store_true",
                   help="スコア >= 0.01 の全推論を score_history.csv に記録")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    available = {n: p for n, p in MODELS.items() if p.exists()}
    if not available:
        print("ERROR: 学習済みモデルが見つかりません。", file=sys.stderr)
        for n, p in MODELS.items():
            print(f"  期待パス: {p}", file=sys.stderr)
        sys.exit(1)

    name_by_stem = {p.stem: n for n, p in available.items()}
    model = WakeWordModel(models=list(available.values()))

    args.out.mkdir(parents=True, exist_ok=True)

    detections_csv = args.out / "detections.csv"
    is_new = not detections_csv.exists()
    csv_fp = open(detections_csv, "a", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_fp)
    if is_new:
        csv_writer.writerow(
            ["id", "datetime", "phrase", "model_stem", "score",
             "wav_filename", "label", "notes"]
        )
        csv_fp.flush()

    score_log_fp = None
    score_log_writer = None
    if args.score_log:
        score_log_csv = args.out / "score_history.csv"
        is_new_sl = not score_log_csv.exists()
        score_log_fp = open(score_log_csv, "a", newline="", encoding="utf-8-sig")
        score_log_writer = csv.writer(score_log_fp)
        if is_new_sl:
            score_log_writer.writerow(["datetime", "model_stem", "score"])
            score_log_fp.flush()

    import pyaudio  # type: ignore[import-untyped]

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )

    pre_frames = int(args.pre * SAMPLE_RATE / FRAME_SAMPLES)
    post_frames = int(args.post * SAMPLE_RATE / FRAME_SAMPLES)
    # バッファは pre + chunk 分を保持（検出時に直近 chunk + その前 pre が取れる）
    buf_capacity = pre_frames + CHUNK_FRAMES
    frame_buffer: deque[np.ndarray] = deque(maxlen=buf_capacity)

    print("=== 検出ロガー開始 ===")
    print(f"閾値: {args.threshold}  pre: {args.pre}s  post: {args.post}s  "
          f"debounce: {args.debounce}s")
    print(f"対象フレーズ: {list(available.keys())}")
    print(f"出力先: {args.out.resolve()}")
    if args.score_log:
        print(f"score_history.csv にも記録 (>= {SCORE_LOG_MIN})")
    print("Ctrl+C で終了\n")

    detection_id = _next_id(detections_csv)
    last_detection_time = 0.0

    try:
        while True:
            data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16)
            frame_buffer.append(frame)

            if len(frame_buffer) < CHUNK_FRAMES:
                continue

            now = time.monotonic()
            if now - last_detection_time < args.debounce:
                continue

            # 直近 CHUNK_FRAMES (2秒) でスコアリング
            recent = list(frame_buffer)[-CHUNK_FRAMES:]
            chunk = np.concatenate(recent)
            scores = model.predict(chunk)

            if score_log_writer:
                ts = datetime.datetime.now().isoformat(timespec="milliseconds")
                for stem, score in scores.items():
                    if score >= SCORE_LOG_MIN:
                        score_log_writer.writerow([ts, stem, f"{score:.4f}"])
                if score_log_fp:
                    score_log_fp.flush()

            best_stem, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score < args.threshold:
                continue

            last_detection_time = now

            # 後続 post_frames 分も録音
            post_data: list[np.ndarray] = []
            for _ in range(post_frames):
                data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                post_frame = np.frombuffer(data, dtype=np.int16)
                post_data.append(post_frame)
                frame_buffer.append(post_frame)

            full_audio = np.concatenate(list(frame_buffer)[: pre_frames + CHUNK_FRAMES]
                                        + post_data)

            detection_id += 1
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_filename = f"{ts_str}_{best_stem}_{best_score:.3f}.wav"
            wav_path = args.out / wav_filename
            sf.write(str(wav_path), full_audio, SAMPLE_RATE, subtype="PCM_16")

            phrase = name_by_stem.get(best_stem, best_stem)
            iso_ts = datetime.datetime.now().isoformat(timespec="seconds")
            csv_writer.writerow(
                [detection_id, iso_ts, phrase, best_stem,
                 f"{best_score:.4f}", wav_filename, "", ""]
            )
            csv_fp.flush()

            print(f"#{detection_id}  ✓「{phrase}」 score={best_score:.3f}  → {wav_filename}")

    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        csv_fp.close()
        if score_log_fp:
            score_log_fp.close()


def _next_id(csv_path: Path) -> int:
    """CSVから最後のidを読んで次のidを返す。新規ファイルなら0。"""
    if not csv_path.exists():
        return 0
    last_id = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].isdigit():
                last_id = max(last_id, int(row[0]))
    return last_id


if __name__ == "__main__":
    main()
