# パウロ先生 国語

国語の文章を読むための学習アプリです。本文に印をつけて質問したり、手書きで文字を入れたりできます。

## つかう

https://meiguangtailang-ux.github.io/paulo-sensei/

インストールはいりません。上のページをブラウザで開くだけです。

## AI について

AI はブラウザの中だけで動きます。読んでいる文章や質問を、外のサーバーに送ることはありません。
はじめて使うときだけ、AI のデータをダウンロードします（Wi‑Fi がおすすめです）。

| はたらき | つかっているもの | 大きさ |
|---|---|---|
| 質問に答える（かしこい） | gemma-2-2b-jpn-it | 約 1.4GB |
| 質問に答える（かるい） | Qwen3-0.6B | 約 350MB |
| 手書きの文字を読む | LT8/japanese-handwriting-onnx | 約 15MB |

「かしこい」が動かない機械では、自動で「かるい」に切りかわります。

## 必要なパソコン

| | 動くことを確かめた最低ライン | おすすめ（目安。まだ確かめていません） |
|---|---|---|
| ブラウザ | Chrome の最新版 | Chrome か Edge の最新版（Mac は Safari 26 でも） |
| グラフィック | WebGPU が使えて、メモリが 4GB（空き約 2GB） | Apple シリコンの Mac、または専用メモリ 4GB 以上 |
| メモリ | 8GB | 16GB |
| 空き容量 | 約 3GB | 約 3GB |

確かめた機械：iMac 2017（Core i5 3.4GHz、メモリ 8GB、Radeon Pro 570 4GB、macOS 13、Chrome）。
この機械で、1回の答えに 4〜13 秒かかります。

iPad は iPadOS 26 の Safari が必要です。実機ではまだ確かめていません。

## ファイル

- `index.html` … アプリ本体（かな漢字変換の辞書も入っています）
- `paulo-ai.js` … AI の部分

## 使わせてもらっているもの

- かな漢字変換の辞書：Mozc の辞書（IPAdic 由来）を小さくしたもの。条件は [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) にあります。
- [WebLLM](https://github.com/mlc-ai/web-llm)（Apache License 2.0）
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)（MIT License）
- gemma-2-2b-jpn-it（Google、[Gemma Terms of Use](https://ai.google.dev/gemma/terms)）
- Qwen3-0.6B（Alibaba Cloud、Apache License 2.0）
- [LT8/japanese-handwriting-onnx](https://huggingface.co/LT8/japanese-handwriting-onnx)（[ETL 文字データベース](https://etlcdb.db.aist.go.jp/)の利用条件）
- 文字：BIZ UDPGothic、BIZ UDPMincho、Bodoni Moda、Shippori Mincho B1、Zen Kaku Gothic New（Google Fonts、SIL Open Font License）

AI のデータはこの場所には置いていません。使うときに配布元からダウンロードします。
