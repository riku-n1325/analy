# LTSラマン校正トムソン散乱解析スターター

このフォルダは、ラマン散乱で絶対値校正したトムソン散乱解析を行うための試作コードです。

現在できることは次の通りです。

- LightField/LightSpeed系の`.spe`ファイルを読み込む
- CCD画像を縦方向に積算して1次元スペクトルにする
- レイリー/ラマンの鋭いピークをガウスフィットする
- 波長ストップで欠けたラマンピークを、欠損範囲を除外して外挿フィットする
- トムソン散乱の広いスペクトル包絡線をフィットする
- ラマン散乱強度を基準に電子密度を計算する
- トムソン散乱幅から電子温度を計算する
- 簡易GUIでファイル選択、条件入力、解析実行を行う

## 1. 簡易アプリの起動

通常は、次のファイルをダブルクリックして起動します。

```text
起動.bat
```

コマンドから起動する場合は、次のように実行します。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\lts_analysis_app.py"
```

GUIでは次を入力します。

- `Thomson SPE`: トムソン散乱スペクトルの`.spe`ファイル
- `Raman SPE`: ラマン散乱スペクトルの`.spe`ファイル
- `Output`: 解析結果CSVなどの保存先
- `Pressure Pa`: 測定時のチャンバー内水素圧力
- `nm/pixel`: 波長校正値。現在の初期値は`0.021`
- `Laser nm`: レーザー波長
- `Angle deg`: 散乱角
- `Raman dσ m2/sr`: ラマン微分散乱断面積
- `Stop min px`, `Stop max px`: 波長ストップで隠れたピクセル範囲

解析ボタンを押すと、電子密度、電子温度、トムソン/ラマンそれぞれのフィット指標`R^2`が表示されます。

## 2. SPEファイルをスペクトル化する

`.spe`を読み込み、2次元CCD画像と1次元スペクトルを出力します。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\read_spe.py" `
  "C:\path\to\data.spe" `
  --out-dir "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter"
```

出力されるファイルは次の通りです。

- `*_preview.png`: 2次元CCD画像のプレビュー
- `*_spectrum.csv`: 縦方向に積算した1次元スペクトル
- `*_spectrum.png`: 1次元スペクトルの簡易プロット

信号がCCDの一部の行だけにある場合は、積算範囲を指定します。

```powershell
--y-min 400 --y-max 650
```

## 3. ラマン/レイリーピークをフィットする

鋭いピークをガウス近似します。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\fit_peak.py" `
  "C:\path\to\data.spe" `
  --out-dir "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter"
```

主なオプションです。

- `--peak-pixel 537`: ピークのおおよその位置を指定する
- `--window 25`: フィット範囲の半幅を指定する
- `--sideband 80`: ベースライン推定に使う左右の範囲を指定する
- `--mask-min 646 --mask-max 654`: 波長ストップで隠れた範囲をフィットから除外する
- `--fixed-center 650`: 欠けたラマンピークの中心を固定する

出力されるファイルは次の通りです。

- `*_fit.png`: 実データとガウスフィットの重ね描き
- `*_fit_curve.csv`: スペクトルとフィット曲線
- `*_fit_summary.csv`: 中心、幅、面積、R^2などのまとめ

波長ストップでラマンピークが欠けている場合は、**直接積分面積ではなくガウス面積を使ってください**。直接積分面積は見えている部分だけの面積なので、本来のラマン信号を過小評価します。

## 4. 仮想ラマンSPEを作る

実ラマンデータがまだない場合、テスト用の仮想ラマン`.spe`を作れます。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\simulate_raman_spe.py" `
  --out "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\virtual_raman.spe"
```

波長ストップで中央が欠けたラマンピークを試す場合は、次のようにします。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\simulate_raman_spe.py" `
  --out "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\virtual_raman_blocked.spe" `
  --stop-min-pixel 646 --stop-max-pixel 654
```

欠けたピークをフィットする例です。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\fit_peak.py" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\virtual_raman_blocked.spe" `
  --out-dir "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter" `
  --fixed-center 650 --peak-pixel 650 --mask-min 646 --mask-max 654 --window 45 --sideband 100
```

## 5. トムソン散乱スペクトルをフィットする

スパイクを含む広いトムソン散乱スペクトルには、専用フィッタを使います。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\fit_thomson.py" `
  "C:\path\to\thomson.spe" `
  --out-dir "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter" `
  --y-min 600 --y-max 970 `
  --fixed-center 536.8749
```

`--fixed-center`には、同じ分光器設定で測ったレイリー線またはレーザー線の中心ピクセルを入れるのが基本です。

出力されるファイルは次の通りです。

- `*_thomson_fit.png`: 生データ、平滑化データ、ベースライン、ガウス包絡線
- `*_thomson_fit_curve.csv`: 生データ、平滑化データ、ベースライン、フィット曲線
- `*_thomson_fit_summary.csv`: 中心、幅、面積、R^2などのまとめ

## 6. ラマン校正による電子密度

トムソン散乱とラマン散乱をそれぞれフィットした後、電子密度を計算します。

```powershell
& "C:\Users\rb_iy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "C:\Users\rb_iy\Documents\Codex\2026-06-30\github\outputs\spe_starter\calibrate_density.py" `
  --thomson-summary "C:\path\to\thomson_fit_summary.csv" `
  --raman-summary "C:\path\to\raman_fit_summary.csv" `
  --pressure-pa 1000 `
  --gas-temperature-k 300 `
  --raman-dsigma 1.0e-34 `
  --scattering-angle-deg 90
```

`--raman-dsigma`には、レーザー波長、ラマン遷移、偏光、集光配置に対応した正しいラマン微分散乱断面積を入れてください。現在の`1.0e-34`は動作確認用の仮値です。

使っている校正式は次です。

```text
ne = n_H2 * (A_TS / A_Raman) * ((dσ/dΩ)_Raman / (dσ/dΩ)_Thomson) * correction_factor
```

ここで、`correction_factor`にはレーザーエネルギー、ゲート幅、検出器ゲイン、分光器透過率、フィルタ透過率などの補正を含められます。

電子温度も同時に出す場合は、波長校正値とレーザー波長を指定します。

```powershell
--nm-per-pixel 0.021 --laser-wavelength-nm 532
```

## 7. フィット評価値について

現在はフィットの評価値として`R^2`を出しています。

- `R^2`が1に近いほど、選んだモデルがデータをよく説明しています。
- ラマンピークのような単峰ガウスでは高い`R^2`が期待できます。
- トムソン散乱ではスパイク、迷光、中心付近の欠損、非ガウス成分があるため、`R^2`は低めになることがあります。

`R^2`は目安であり、最終判断では必ずフィット図も確認してください。
