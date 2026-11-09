# GB10 (`sm_121`) 向け `comfy-kitchen` フォーク — 索引

English: [SM121_FORK_GUIDE.en.md](SM121_FORK_GUIDE.en.md)

上流 `Comfy-Org/comfy-kitchen` のフォーク。GB10 (sm_121) 向けに **INT8 GEMM の Triton 経路**と
**変調（scale/shift）の融合**を追加したもので、実体は `gb10-optimized` ブランチの7コミットです。
この文書は用途別に4本へ分割しています（旧 `SM121_FORK_GUIDE.md` 1本を再編）。

| 項目 | 値 |
| :--- | :--- |
| 対象機 | dgxspark1（GB10 / 48 SM / 2190 MHz、クロック上限 2200） |
| 前提 | CUDA 13.x、Python 3.13、`torch 2.15+cu132`、Triton 3.9 |
| 版数 | `0.2.41+gb10.1`（PEP 440 のローカル版数。上流版との取り違え防止） |
| 変更範囲 | **Python のみ**（CUDA/C++ の差分なし。`_C.abi3.so` は再ビルドするが中身は同一） |
| 既定ブランチ | `gb10-optimized`（`main` は上流ミラー） |

---

## 0. 文書マップ

| 知りたいこと | 文書 | 章 |
| :--- | :--- | :--- |
| まず動かす／入れ直す | [OPERATIONS.md](docs/gb10/OPERATIONS.md) | §3〜§5 |
| ブランチ運用・別マシン再現 | OPERATIONS.md | §2 |
| 上流への追従（rebase） | OPERATIONS.md | §6 |
| 障害対応・ロールバック | OPERATIONS.md | §7〜§8 |
| 何をどう変えたか（設計根拠） | [DESIGN.md](docs/gb10/DESIGN.md) | §1〜§7 |
| やらない最適化・レビュー記録・既知の制限 | DESIGN.md | §8〜§10 |
| 実測値・再測定の手順 | [MEASUREMENTS.md](docs/gb10/MEASUREMENTS.md) | §1〜§10 |
| 謝辞・出典 | DESIGN.md | §11 |

---

## 1. 最短手順（詳細は OPERATIONS §3〜§5）

```bash
cd ~/comfy-kitchen
git submodule update --init --depth 1
source ~/ComfyUI/.venv/bin/activate          # ComfyUI に読ませる venv（ノードごとにパスが違う）
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q      # 290 passed
git -C ~/ComfyUI apply ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch
# ComfyUI を再起動して反映
```

---

## 2. フォークの状態（要点）

- 既定ブランチは `gb10-optimized`。フォークの実体は `main..gb10-optimized` の7コミットで、
  `git log --oneline main..origin/gb10-optimized` が正（SHA は rebase ごとに変わる）。
- `main` は上流と同一コミットのみのミラー。**このクローンで `main` を checkout しない** —
  `~/comfy-kitchen` は ComfyUI の venv に editable で入っているため、切り替えた瞬間に
  ComfyUI が素のカーネルで動き出す（しかも無言で）。ミラー更新は checkout を伴わない
  `git branch -f main upstream/main`（OPERATIONS §2）。
- 上流が同等機能（sm_121 判定・融合カーネル）を入れたら、**自分のゲートを捨てて上流に従う**。
  二重ディスパッチは事故のもと（OPERATIONS §6）。

---

## 3. 変更点の要約

| ファイル | 変更 | 効果 | 詳細 |
| :--- | :--- | :--- | :--- |
| `comfy_kitchen/backends/triton/quantization.py` | 融合 INT8 GEMM + dequant epilogue、M帯別タイル表、`_ensure_python_headers()` のメモ化 | 広い N の射影層で 1.5〜2.4x | DESIGN §2 |
| `comfy_kitchen/backends/cuda/__init__.py` | sm_121 ディスパッチ（`_TRITON_INT8_MIN_M = 40`）、モード読取、CUTLASS フォールバック | 上と同じ経路を ComfyUI から使う | DESIGN §3 |
| `comfy_kitchen/__init__.py` | `torch.compiler.is_compiling()` 時の早期リターン | `torch.compile` 下で INT8 が落ちる問題を回避 | DESIGN §4 |
| `comfy_kitchen/tensor/scale_shift.py`（新規） | per-segment 変調を1パスで実行（ビット一致） | 変調段階で 3.0x | DESIGN §5 |
| `tests/test_int8.py`, `tests/test_scale_shift.py` | ビット一致・タイル・エッジ形状・残差の回帰 | 変更時の安全網 | DESIGN §6 |
| `patches/comfyui-minimax-modulation.patch` | ComfyUI 側フック（`_mod_scale_shift`） | 実運用でのみ効く | DESIGN §7 |

---

## 4. 実測サマリ

H3（M=15,787）の1イテレーション: **11.4 s → 7.0 s（−4.4 s / 1.63x）**。内訳は INT8 GEMM
7.87 → 3.82 s、変調 0.49 → 0.16 s、回転・量子化と attention は不変。attention の差はフォークではなく
ComfyUI の起動フラグ `--use-sage-attention` によるもの。

効く条件は **N が広い**射影層（N ≳ 8192）で、N ≤ 4096 では 1.05〜1.06x。数値・測定条件・再測定手順は
[MEASUREMENTS.md](docs/gb10/MEASUREMENTS.md)。

---

## 5. 触る前の原則

- 変更したら OPERATIONS §5 の3点（ディスパッチとビット一致／タイル表／pytest）を必ず流す。
- 測定は min-of-3〜5、構成はラウンドロビン。**10% 未満の差は有意とみなさない**（実行間で 10〜15% 揺れる）。
- 「効かないと確認済み」の案を再試行しない（DESIGN §8）。
