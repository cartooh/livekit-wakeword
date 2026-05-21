"""マイクでリアルタイム日本語コマンド検出テスト。

使い方:
    uv run python examples/mic_test_japanese.py [--threshold 0.1]

引数:
    --threshold   検出スコアの閾値 (0〜1, デフォルト: 0.1)
                  低いほど検出しやすいが誤検出も増える
                  高いほど厳しくなり見逃しが増える
"""

import argparse
import asyncio
from pathlib import Path

from livekit.wakeword import WakeWordModel
from livekit.wakeword.inference import WakeWordListener

# モデルパスをフレーズ名でマッピング
MODELS = {
    "止まって下さい": Path("output/tomatte_kudasai/tomatte_kudasai.onnx"),
    "こっちに来て":   Path("output/kocchi_ni_kite/kocchi_ni_kite.onnx"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="日本語コマンドのリアルタイム検出テスト")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="検出スコアの閾値 (デフォルト: 0.1)",
    )
    p.add_argument(
        "--debounce",
        type=float,
        default=1.5,
        help="同じコマンドの再検出を無視する秒数 (デフォルト: 1.5)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # 存在するモデルだけ読み込む
    available: dict[str, Path] = {
        name: path for name, path in MODELS.items() if path.exists()
    }

    if not available:
        print("エラー: モデルが見つかりません。")
        print("先にトレーニングとエクスポートを完了させてください:")
        for name, path in MODELS.items():
            print(f"  {name}: {path}")
        return

    print("=== 日本語コマンド検出テスト ===")
    print(f"閾値: {args.threshold}  デバウンス: {args.debounce}s")
    print("検出対象:")
    for name, path in available.items():
        print(f"  「{name}」← {path}")
    if missing := set(MODELS) - set(available):
        print("未学習のため除外:")
        for name in missing:
            print(f"  「{name}」← {MODELS[name]} (not found)")
    print("\nマイクに向かって話しかけてください。Ctrl+C で終了。\n")

    model = WakeWordModel(models=list(available.values()))

    async with WakeWordListener(
        model,
        threshold=args.threshold,
        debounce=args.debounce,
    ) as listener:
        while True:
            detection = await listener.wait_for_detection()
            # モデルファイル名からフレーズ名を逆引き
            phrase = next(
                (name for name, path in available.items() if path.stem == detection.name),
                detection.name,
            )
            print(f"✓ 検出:「{phrase}」  スコア={detection.confidence:.3f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n終了しました。")
