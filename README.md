# Weather-Based Sales Prediction System

## 概要 (Overview)
このプロジェクトは、天気データに基づいて売上を予測するシステムです。気温、天気状態、不快指数、スタッフ数などの要因が売上にどのように影響するかを分析し、最小二乗法を用いて将来の売上を予測します。

## 機能 (Features)
- CSV形式のデータからの統計量計算
- 欠損値の適切な処理
- データの正規化処理
- 最小二乗法による売上予測
- 視覚的なグラフによる結果の表示

## 必要条件 (Requirements)
- Python 3.6以上
- pandas
- numpy
- matplotlib

## ファイル構成 (File Structure)
- `weather_sales_analyzer.py` - メインプログラム
- `sales_learn_data.csv` - 学習用データセット
- `sales_predict_data.csv` - 予測用データセット
- `sales_prediction_results.png` - 生成される予測結果グラフ
- `sales_factors_analysis.png` - 生成される要因分析グラフ

## 使用方法 (Usage)

### データの準備
学習用および予測用のCSVファイルは以下の列を含む必要があります：
- `date` - 日付 (YYYY-MM-DD形式)
- `sales` - 売上金額（予測データでは欠損値でも可）
- `weather` - 天気 (0: 快晴, 1: 晴れ, 2: 曇り, 3: 雨)
- `temperature` - 最高気温 (°C)
- `THI` - 不快指数
- `num_staff` - スタッフ数

### コマンド
```bash
# ステップ1: 統計量の計算
python weather_sales_analyzer.py step1 sales_learn_data.csv

# ステップ2: 欠損値処理と正規化
python weather_sales_analyzer.py step2 sales_learn_data.csv

# ステップ3: 売上予測とグラフ生成
python weather_sales_analyzer.py step3 sales_learn_data.csv sales_predict_data.csv
