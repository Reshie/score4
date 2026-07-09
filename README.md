# score4-alphazero

AlphaZero 方式で立体四目並べ（4x4x4 Score Four）を自己対局学習するための最小実装です。

## ルール

- 盤面は `x=4, y=4, z=4` の 64 マスです。
- 行動は 4x4 の柱を 1 つ選ぶ `0..15` の 16 通りです。
- 石は選んだ柱の最下段から積まれます。
- 縦、横、斜め、空間対角線を含む全 76 本のラインのどれかで 4 つ並べると勝ちです。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[train]"
```

PyTorch のインストール方法を細かく指定したい場合は、先に公式手順で `torch` を入れてから `python -m pip install -e .` を実行してください。

## テスト

```powershell
python -m unittest discover -s tests
```

## 学習の試運転

CPU でも動作確認しやすい小さめの設定です。

```powershell
python -m score4.train --iterations 2 --games-per-iteration 4 --simulations 32 --train-steps 20 --batch-size 32
```

本格的に回す場合は、`--games-per-iteration`、`--simulations`、`--train-steps`、`--channels`、`--res-blocks` を増やしてください。

```powershell
python -m score4.train --iterations 100 --games-per-iteration 64 --simulations 200 --train-steps 200 --batch-size 128 --channels 96 --res-blocks 6
```

チェックポイントは既定で `runs/score4/checkpoint_XXXX.pt` に保存されます。途中から再開する場合:

```powershell
python -m score4.train --resume runs/score4/checkpoint_0010.pt
```

学習中は自己対局と train step の進捗バーが表示されます。終了後は同じディレクトリに `metrics.csv` と `training_progress.svg` が保存され、loss や先手スコアの推移を確認できます。

自己対局は既定で複数ゲームを同時に進める batch MCTS を使い、network 評価をまとめて GPU に送ります。並列ゲーム数は `--self-play-batch-size` で指定でき、既定は `32` です。従来の 1 ゲームずつの自己対局に戻す場合は `--self-play-batch-size 1` を指定してください。

network 評価キャッシュも既定で有効です。メモリを抑えたい場合は `--eval-cache-size 0` を指定してください。MCTS の探索木再利用も試したい場合は `--reuse-tree` を追加できます。

## モデルと対戦

学習済み checkpoint と対戦できます。

```powershell
python -m score4.play --checkpoint runs/score4/checkpoint_0002.pt --simulations 100
```

入力は `0..15` の列番号、または `x y` の座標です。後手で遊ぶ場合:

```powershell
python -m score4.play --checkpoint runs/score4/checkpoint_0002.pt --human second
```

探索を強くしたい場合は `--simulations` を増やしてください。

## 構成

- `src/score4/game.py`: 盤面、合法手、勝利判定、入力エンコード
- `src/score4/mcts.py`: PUCT MCTS
- `src/score4/model.py`: 3D 畳み込みの policy/value ネットワーク
- `src/score4/self_play.py`: 自己対局データ生成
- `src/score4/train.py`: 学習 CLI
- `src/score4/play.py`: 学習済みモデルとの対戦 CLI
- `tests/`: 盤面と MCTS のスモークテスト
