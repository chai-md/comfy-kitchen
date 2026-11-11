# GB10 フォーク運用 — ビルド・検証・上流追従・障害対応

English: [OPERATIONS.en.md](OPERATIONS.en.md)

対象: `chai-md/comfy-kitchen` の `gb10-optimized` ブランチ。設計根拠は
[DESIGN.md](DESIGN.md)、実測は [MEASUREMENTS.md](MEASUREMENTS.md) を参照。

---

## 1. 前提

- 対象機: dgxspark1（GB10 / 48 SM / 2190 MHz、クロック上限 2200）
- CUDA 13.x、Python 3.13、`torch 2.15+cu132`、Triton 3.9
- 作業は **`source .venv/bin/activate` した venv の中で行う**（§3.2）。ノードごとに venv のパスが違うので、
  手順に絶対パスを埋めない。

---

## 2. フォークとブランチ

| 項目 | 実態 |
| :--- | :--- |
| フォーク | `chai-md/comfy-kitchen`（`origin`、HTTPS）。**default branch は `gb10-optimized`** |
| 上流 | `Comfy-Org/comfy-kitchen`（`upstream`、`git fetch upstream` で追従） |
| 作業ブランチ | **`gb10-optimized`**（7コミット＝パッチ系列5＋docs/bench 1＋版数1。常に `main` の直上に rebase する） |
| `main` | **上流のミラー**（上流と同一コミットのみ）。直接コミットしない・checkout しない |
| 変更の実体 | `git log --oneline main..origin/gb10-optimized` の 7 コミット |
| ローカル | このリポジトリ。ビルド/テストは直下の `.venv`、ComfyUI から使う場合は ComfyUI 側の venv にも editable で入れる（§3.3） |

```bash
# フォーク作成 + clone（初回のみ）
gh repo fork Comfy-Org/comfy-kitchen --clone
cd comfy-kitchen
git remote add upstream https://github.com/Comfy-Org/comfy-kitchen.git
git fetch upstream
git switch -c gb10-optimized --track origin/gb10-optimized   # 既存フォークから引く場合
git submodule update --init --depth 1   # 必須（§3.1）
```

**`main` は checkout しない。** `~/comfy-kitchen` は ComfyUI の venv に editable で入っているので、
このクローンで `main` に切り替えた瞬間に ComfyUI が素のカーネルで動き出す（しかも無言で）。
ミラーの更新は checkout を伴わないこの形で行う:

```bash
git fetch upstream && git branch -f main upstream/main && git push origin main
```

履歴は 7 本（パッチ系列5＋docs/bench 1＋版数1）。**ブランチ名ではなく範囲で追う**（`main..gb10-optimized` がフォークの実体）。

| コミット | 内容 |
| :--- | :--- |
| `f5d2558` | Triton INT8 融合 GEMM＋タイル表＋ディスパッチ＋residual 融合＋`tensor/scale_shift.py` 初版＋テスト |
| `dfcd33d` | `scale_shift`: 変調テンソルを活性 dtype に丸める（fp32 adaLN 対応。これが無いと H3 の絵が変わる） |
| `b319605` | `scale_shift`: 非連続な変調ビューを `.contiguous()` 化、カバレッジ検証を追加 |
| `2ce8f0d` | ComfyUI 側フックを `patches/comfyui-minimax-modulation.patch` として同梱 |
| `0f83d79` | 比較基準を素の comfy-kitchen に修正、Triton ゲートを 40 へ、計測スクリプト `bench/` を同梱 |
| `e0aa594` | guide / bench / docs の集約1本（旧11本を squash。元メッセージ全文は commit body に保存） |
| `400aee0` | `pyproject.toml` の版数を `0.2.41+gb10.1` に（上流版との取り違え防止） |

版数の `+gb10.1` は PEP 440 のローカル版数で、既定ブランチから
`pip install git+https://github.com/chai-md/comfy-kitchen` した場合にこの値になる。既存の editable 環境は
再インストールするまで dist-info が `0.2.41` のまま（import 先は変わらないので動作に影響はない）。

> 注: rebase のたびに SHA は書き換わる。上の SHA は 2026-11-09 時点。**範囲と内容で追う**こと。

```bash
# 同じ状態を別マシンで再現する場合
git fetch origin && git switch gb10-optimized && git reset --hard origin/gb10-optimized
```

`sm121_optimize.patch` は `git diff` のスナップショットで、**リポジトリには置かない**（`~/archive/patches/`）。
必要なら `git diff main..gb10-optimized -- comfy_kitchen tests > ~/archive/patches/sm121_optimize.patch`。

---

## 3. ビルドとインストール

### 3.1 サブモジュール（先に必ず）

```bash
cd ~/comfy-kitchen
git submodule update --init --depth 1
```

空のままビルドすると 2 つの事故が起きます。

1. `flash_decode.cu` が `flash_attn` ヘッダを見つけられずビルドが途中で失敗する
2. CMake が `COMFY_HAVE_CUTLASS` を無効化し、`cutlass_gemm_int8.cu` 等が **4KB のダミースタブ**として
   コンパイルされる → ビルドは成功するが遅い状態になる

初期化前に一度ビルドしていた場合は `rm -rf build` で CMake キャッシュを捨ててください。

### 3.2 ビルド（editable）

前提: その venv に **torch（2.15+cu132）と triton（3.9）が入っていること**。
`--no-build-isolation` はビルドに既存の torch を使うため、空の venv では失敗する。

```bash
cd ~/comfy-kitchen                 # ノードごとのリポジトリ位置

# リポジトリ直下に .venv がある場合
source .venv/bin/activate
# 無い場合は、そのノードの ComfyUI の venv を使う（torch/triton が揃っている）
#   source ~/ComfyUI/.venv/bin/activate

which python                      # venv を指していることを確認
uv pip install nanobind packaging wheel cmake ninja   # 不足時のみ
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
```

まっさらな venv から始める場合だけ、先に依存を入れる（数 GB・数分）。

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U uv
uv pip install torch triton        # ComfyUI と同じバージョンに合わせる
```

- `COMFY_CUDA_ARCHS=121` で旧世代 arch を省略（20 コアで約 2.5 分）
- `-e`（editable）なので `_C.abi3.so` はリポジトリ内に置かれ、以後 **Python の修正は再ビルド不要**
- 確認: `ls -l comfy_kitchen/backends/cuda/_C.abi3.so` が 12MB 級であること（4KB なら §3.1）
- `uv pip install` は **activate 中の venv に入る**（`VIRTUAL_ENV` を見る）。空のまま実行すると
  別インタプリタに入って「import できない」になるので、`which python` で `.venv` を指しているか確認する
- この venv に `pip` モジュールが無い構成でも `uv pip` は動く。`python -m pip` は使わない

### 3.3 ComfyUI の venv へ入れる

editable は **venv ごと**に入る。検証だけなら §3.2 の `.venv` で足りるが、ComfyUI に読ませるなら
そのノードの ComfyUI の venv を activate して同じコマンドを流す（`--python` を足すより事故が少ない）。

```bash
source ~/ComfyUI/.venv/bin/activate      # ノードごとのパス
cd ~/comfy-kitchen
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -c "import comfy_kitchen, pathlib; print(pathlib.Path(comfy_kitchen.__file__).parent)"
# → <repo>/comfy_kitchen なら editable が効いている
```

再ビルド不要なのは Python 側だけ。**C++/CUDA（`_C.abi3.so`）を触ったら両方の venv で再ビルド**する。

### 3.4 インストール直後の動作確認

```bash
python -c "
import torch
from comfy_kitchen.backends import cuda as cb
import comfy_kitchen.tensor.scale_shift as ss
print('cap', cb._cuda_device_capability(0), 'triton-int8', cb._should_use_triton_int8_dequant(0))
h = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
s = torch.randn(2, 1024, dtype=torch.float32, device='cuda')
b = torch.randn(2, 1024, dtype=torch.float32, device='cuda')
segs = [(0, 512, 0), (512, 1024, 1)]
out = ss.scale_shift(h, s, b, ss.row_mod_from_segments(1024, segs, h.device))
ref = h.clone()
for a, z, r in segs:
    ref[a:z].mul_(1.0 + s[r].to(h.dtype)).add_(b[r].to(h.dtype))
print('scale_shift bit-identical:', torch.equal(out, ref))
"
```

期待値: `cap (12, 1) triton-int8 True` / `scale_shift bit-identical: True`。

---

## 4. ComfyUI 側フックの適用（任意）

ComfyUI は上流（`Comfy-Org/ComfyUI`）クローンなので push 先が無い。フックは本リポジトリの
`patches/comfyui-minimax-modulation.patch` を適用して使う。

**2026-11-11 以降、既定では適用しない**（効果 0.28 s/forward、維持コストが上回る。DESIGN §7）。
適用する場合のみ以下を実行する。

```bash
git -C ~/ComfyUI apply ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 取り消し（上流に戻す）
git -C ~/ComfyUI apply -R ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 上流を pull した後は再適用（競合したら上流の追加行は残したまま、本パッチの import ブロックを差し込む）
```

本パッチは ComfyUI `0228a9d73`（v0.45.0）向けに更新済み。上流 #17655（H3-image 対応）で
`_tag_runs` の直後に import ブロックが入る形に変わっており、旧版のままでは `model.py` で衝突する。

### 4.1 H3 コンテキストウィンドウ修正（別パッチ・任意）

`model.py` の `PackedLayout` と `model_base.py` の `_window_payload` に手を入れる修正は、
フックとは別パッチ（`~/archive/patches/comfyui-h3-contextwindow-*.diff`）で管理している。
**これも既定では適用しない**。コンテキストウィンドウ（10 秒超の分割生成）を使う場合だけ必要で、
単発生成しかしないなら不要（ウィンドウ生成自体が step 時間 +47% で、継続性も完全ではない）。
適用する場合はフック適用後に同じ要領で適用し、`check_h3_window_positions.py` で位置
（video err ≲1e-9）を確認する。

```bash
git -C ~/ComfyUI apply ~/archive/patches/comfyui-h3-contextwindow-20261111.diff
python <skill>/scripts/check_h3_window_positions.py     # OK - windows sit at their real clip position
```

反映には **ComfyUI の再起動**が必要（Python の読み直しのみ）。

---

## 5. 検証（変更したら必ず全部）

### 5.1 ディスパッチとビット一致

```bash
python -c "
import torch, comfy_kitchen
from comfy_kitchen.backends import cuda as cb
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE
from comfy_kitchen.backends.triton.quantization import triton_int8_gemm_dequant

print('cap', cb._cuda_device_capability(0), 'dispatch', cb._should_use_triton_int8_dequant(0))
assert cb._should_use_triton_int8_dequant(0) is True

for m in (24, 64, 96, 160, 192, 256, 512):
    n, k = 256, 256
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    xs = torch.rand(m, 1, dtype=torch.float32, device='cuda') * 0.02
    ws = torch.rand(n, dtype=torch.float32, device='cuda') * 0.02
    bias = cb._empty_cuda_tensor(x.device, torch.bfloat16)
    a = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    b = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    assert triton_int8_gemm_dequant(x, w, xs, ws, bias, a)
    assert cb._C.cutlass_int8_dequant(cb._wrap_for_dlpack(x), cb._wrap_for_dlpack(w),
        cb._wrap_for_dlpack(xs), cb._wrap_for_dlpack(ws), cb._wrap_for_dlpack(bias),
        cb._wrap_for_dlpack(b), DTYPE_TO_CODE[torch.bfloat16],
        torch.cuda.current_stream().cuda_stream)
    assert torch.equal(a, b), f'M={m} mismatch'
    print('M', m, 'bit-identical OK')
"
```

### 5.2 タイル表

```bash
python -c "
from comfy_kitchen.backends.triton.quantization import _int8_tile_config as f
for m in (24, 64, 127, 128, 207, 208, 287, 288, 4096):
    print(m, f(m, 4096, 4096)[:2], '|', m, f(m, 5120, 5120)[:2])
"
```

期待値: `24..127 → (64,64)` / `128..207 → (64,128)` / `208..287 → (128,64)` / `288.. → (128,128)`、
`5120×5120` は全て `(128,128)`。

### 5.3 単体テスト

```bash
uv pip install pytest        # 未導入なら
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
```

期待値: 全件パス（本リポジトリでは `290 passed` = 上流 + 追加 96 ケース）。
`test_scale_shift.py` は非連続ビュー・subnormal・カバレッジ欠落も落とすので、
カーネルを触ったら必ず両方流すこと。実測の取り直しは [MEASUREMENTS.md](MEASUREMENTS.md) §10。

---

## 6. 上流追従

作業ブランチは `gb10-optimized`（§2）。上流の新リリースに載せ替えるときはこの順で。

```bash
cd ~/comfy-kitchen
git fetch upstream
git switch gb10-optimized                   # main は checkout しない（§2）
git rebase upstream/main                    # パッチ列（7本）を上流 tip に載せ替え
git submodule update --init --depth 1

source .venv/bin/activate                   # §3.2 の venv
COMFY_CUDA_ARCHS=121 uv pip install --no-build-isolation -e .
python -m pytest tests/test_int8.py tests/test_scale_shift.py -q
git push --force-with-lease origin gb10-optimized

# main ミラーを上流へ（fast-forward。checkout 不要）
git branch -f main upstream/main && git push origin main
```

- GitHub の "Sync fork" は **`main` を開いている時だけ**押す（`gb10-optimized` には上流の同名ブランチが無いので出ない）。
  `gh repo sync` も既定ブランチを見るため、使うなら `gh repo sync chai-md/comfy-kitchen -b main` と明示する。
- 同期点にタグを打つなら `gb10-0.2.41-1` のように **`v` prefix を避ける**
  （fork の `build-wheels.yml` が `tags: v*` で全ターゲットのホイールビルドを起動する）。
- コンフリクトしやすいのは **(1) `_cutlass_int8_dequant()` の 3 呼び出し箇所**、
  **(2) triton 側 `int8_linear()` の末尾**、**(3) `__init__.py` の `int8_linear()` 冒頭**。
- 上流が同等機能（sm_121 判定や融合カーネル）を入れた場合は、**自分のゲートを捨てて上流に従う**。
  二重ディスパッチは事故のもと。
- 追従後は §5 を必ず再実行（特に 5.1 のビット一致。上流が epilogue の演算順序を変えたら一致しなくなる）。
- フォークを ComfyUI から使う場合、`pip install` の向き先が fork のパスであることを確認（§3.3 の確認コマンド）。
- 上流が `_TRITON_INT8_MIN_M` の帯域やタイル表に触れたら、[MEASUREMENTS.md](MEASUREMENTS.md) の比較表を取り直す
  （`~/archive/patches/bench_int8_baseline.py` を A / `gb10-optimized` の 2 状態で流すだけ）。
- ComfyUI 側は `git pull` でフックが外れるので、§4 のパッチを再適用する。

---

## 7. トラブルシューティング

| 症状 | 原因 / 対処 |
| :--- | :--- |
| `ModuleNotFoundError: comfy_kitchen` | editable の向き先（リポジトリ）が消えた。§3 を再実行 |
| ビルドは通るが遅い（CUTLASS と変わらない） | サブモジュール未初期化で `cutlass_gemm_int8.cu` がダミー化。`rm -rf build` → §3.1 から |
| `Kernel requires a runtime memory allocation, but no allocator was set` | TMA 記述子を使う実験時のみ。`triton.set_allocator(...)` を設定 |
| `stderr` に `Triton INT8 GEMM failed ... falling back to CUTLASS` | `_triton_int8_broken` が立ち、そのセッションは CUTLASS 固定。原因（多くは Python.h 不在や Triton のコンパイル失敗）を直してプロセス再起動 |
| `Python.h` が見つからない | `_ensure_python_headers()` のフォールバック（`/opt/py312-include`、コンテナ用）。ホスト venv では `sysconfig` の include が使われる |
| 数値が CUTLASS と一致しない | ランチャーの `BM/BN/BK/G/warps/stages` を定数のままにしている（DESIGN §2）。ただし **N が16の倍数でない場合は上流 CUTLASS の方が壊れている**（DESIGN §9.4）ので、厳密参照で判定する |
| `expand()` したスケールで値が壊れる / OOB | ゲートが `is_contiguous()` を要求する。呼び出し側で `.contiguous()` するか、CUTLASS に落とす |
| `torch.compile` で autograd エラー | `requires_grad` 付き入力は意図的に Python trace へ落とす（DESIGN §9.1）。訓練で使うなら custom op に autograd 登録が必要 |
| ComfyUI 起動ログの `comfy-kitchen version` が `0.2.41` のまま | §3.3 の再インストールをしていない（`0.2.41+gb10.1` になるはず） |

---

## 8. ロールバック（素の comfy-kitchen に戻す）

```bash
# 1) 実行時のみ: ディスパッチを切る（プロセス再起動で戻る）
COMFY_KITCHEN_INT8_MODE=cutlass python main.py ...

# 2) 常時: モードファイル（/etc/comfy-kitchen-int8.mode に off / cutlass を書く）

# 3) ComfyUI 側フックだけ戻す
git -C ~/ComfyUI apply -R ~/comfy-kitchen/patches/comfyui-minimax-modulation.patch

# 4) フォークごと戻す（上流相当の状態）
source ~/ComfyUI/.venv/bin/activate
uv pip install --force-reinstall "comfy-kitchen==0.2.41"
```

注意: `comfy-kitchen==0.2.41` は PEP 440 の public 版一致で `0.2.41+gb10.1` にも一致するため、
`--force-reinstall` を付けないと fork の editable が残る。
