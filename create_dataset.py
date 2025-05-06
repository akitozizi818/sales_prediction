import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# シード値を設定して再現性を確保
np.random.seed(42)
random.seed(42)

# 学習データセットの作成 (2023年の1月から6月まで)
def create_learn_dataset(num_rows=180):
    # 日付の作成 (2023年1月1日から)
    start_date = datetime(2023, 1, 1)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_rows)]
    
    # 天気の作成 (0: 快晴, 1: 晴れ, 2: 曇り, 3: 雨)
    weathers = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.1, 0.4, 0.3, 0.2])
    
    # 最高気温の作成 (季節変動を含む)
    base_temps = np.linspace(5, 25, num_rows)  # 季節による気温上昇
    random_variation = np.random.normal(0, 3, num_rows)  # ランダム変動
    temperatures = base_temps + random_variation
    
    # 不快指数の作成 (気温との相関と季節変動)
    base_thi = temperatures * 1.8 + np.random.normal(0, 5, num_rows)
    thi = np.clip(base_thi, 40, 85)  # 不快指数の範囲を制限
    
    # スタッフ数の作成 (週末は多め)
    base_staff = np.ones(num_rows) * 5
    weekend_mask = [i % 7 in [5, 6] for i in range(num_rows)]  # 土日判定
    base_staff[weekend_mask] += 2  # 週末は基本+2人
    staff_variation = np.random.randint(-1, 2, num_rows)  # -1, 0, 1のランダム変動
    num_staff = base_staff + staff_variation
    
    # 売上の作成 (天気、気温、不快指数、スタッフ数に依存)
    base_sales = 50000 + temperatures * 1000  # 気温が高いほど売上増加
    
    # 天気による影響 (快晴・晴れで売上増加、雨で減少)
    weather_effect = np.zeros(num_rows)
    weather_effect[weathers == 0] += 15000  # 快晴
    weather_effect[weathers == 1] += 10000  # 晴れ
    weather_effect[weathers == 2] += 0      # 曇り
    weather_effect[weathers == 3] -= 10000  # 雨
    
    # スタッフ数による影響
    staff_effect = (num_staff - 5) * 5000
    
    # 不快指数による影響 (適度な不快指数が売上に良い)
    thi_effect = -((thi - 65) ** 2) * 10  # 65が最適
    
    # 曜日効果 (週末に売上増加)
    weekend_effect = np.zeros(num_rows)
    weekend_effect[weekend_mask] += 20000
    
    # すべての効果を合計し、ノイズを追加
    noise = np.random.normal(0, 5000, num_rows)
    sales = base_sales + weather_effect + staff_effect + thi_effect + weekend_effect + noise
    
    # 欠損値の導入 (約5%)
    missing_indices = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
    sales[missing_indices[:len(missing_indices)//3]] = np.nan
    temperatures[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3]] = np.nan
    thi[missing_indices[2*len(missing_indices)//3:]] = np.nan
    
    # データフレームの作成
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'weather': weathers,
        'temperature': temperatures,
        'THI': thi,
        'num_staff': num_staff
    })
    
    return df

# 予測データセットの作成 (2023年7月の2週間)
def create_predict_dataset(num_rows=14):
    # 日付の作成 (2023年7月1日から)
    start_date = datetime(2023, 7, 1)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_rows)]
    
    # 天気の作成
    weathers = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.15, 0.45, 0.25, 0.15])
    
    # 夏の最高気温の作成
    temperatures = np.random.normal(30, 3, num_rows)  # 7月なので高め
    
    # 夏の不快指数の作成
    thi = temperatures * 1.8 + np.random.normal(20, 3, num_rows)  # 夏なので不快指数高め
    thi = np.clip(thi, 60, 90)  # 夏の不快指数範囲
    
    # スタッフ数の作成
    base_staff = np.ones(num_rows) * 6  # 夏休みなどで基本スタッフ数を増やす
    weekend_mask = [i % 7 in [5, 6] for i in range(num_rows)]
    base_staff[weekend_mask] += 2
    staff_variation = np.random.randint(-1, 2, num_rows)
    num_staff = base_staff + staff_variation
    
    # データフレームの作成 (予測データには売上にNaNを設定)
    sales = np.full(num_rows, np.nan)  # すべてNaNの配列を作成
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,  # 空の売上列を追加
        'weather': weathers,
        'temperature': temperatures,
        'THI': thi,
        'num_staff': num_staff
    })
    
    return df

# 学習データと予測データを作成
learn_data = create_learn_dataset()
predict_data = create_predict_dataset()

# CSVファイルとして保存
learn_data.to_csv('sales_learn_data.csv', index=False)
predict_data.to_csv('sales_predict_data.csv', index=False)

# データの確認
print("学習データサンプル:")
print(learn_data.head())
print("\n学習データの統計情報:")
print(learn_data.describe())

print("\n予測データサンプル:")
print(predict_data.head())
print("\n予測データの統計情報:")
print(predict_data.describe())

# 学習データに含まれる欠損値の確認
print("\n学習データの欠損値数:")
print(learn_data.isna().sum())

# 予測データに含まれる欠損値の確認
print("\n予測データの欠損値数:")
print(predict_data.isna().sum())