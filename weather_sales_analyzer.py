import pandas as pd
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

def read_csv_file(file_path):
    """CSVファイルを読み込み、データフレームとして返す関数"""
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(file_path)
        
        # カラム名を正規化する
        df.columns = [col.strip() for col in df.columns]
        
        return df
    except Exception as e:
        print(f"CSVファイルの読み込みでエラーが発生しました: {e}")
        sys.exit(1)

def step1(file_path):
    """Step 1: CSVファイルから統計量を計算する関数"""
    # データを読み込む
    df = read_csv_file(file_path)
    
    # 必要な列を確認する
    required_cols = ['date', 'sales', 'weather', 'temperature', 'THI', 'num_staff']
    
    # 欠損値を除外せずに統計量を計算する
    sales_stats = {
        'mean': df['sales'].mean(),
        'var': df['sales'].var(),
        'min': df['sales'].min(),
        'max': df['sales'].max()
    }
    
    temp_stats = {
        'mean': df['temperature'].mean(),
        'var': df['temperature'].var(),
        'min': df['temperature'].min(),
        'max': df['temperature'].max()
    }
    
    thi_stats = {
        'mean': df['THI'].mean(),
        'var': df['THI'].var(),
        'min': df['THI'].min(),
        'max': df['THI'].max()
    }
    
    staff_stats = {
        'mean': df['num_staff'].mean(),
        'var': df['num_staff'].var(),
        'min': df['num_staff'].min(),
        'max': df['num_staff'].max()
    }
    
    # 結果を指定された形式で出力する
    print(f"{sales_stats['mean']},{sales_stats['var']},{sales_stats['min']},{sales_stats['max']}")
    print(f"{temp_stats['mean']},{temp_stats['var']},{temp_stats['min']},{temp_stats['max']}")
    print(f"{thi_stats['mean']},{thi_stats['var']},{thi_stats['min']},{thi_stats['max']}")
    print(f"{staff_stats['mean']},{staff_stats['var']},{staff_stats['min']},{staff_stats['max']}")
    
    return sales_stats, temp_stats, thi_stats, staff_stats

def step2(file_path, sales_stats, temp_stats, thi_stats, staff_stats):
    """Step 2: 欠損値の処理と正規化を行う関数"""
    # データを読み込む
    df = read_csv_file(file_path)
    
    # 「売上金額」の欠損値を平均値で埋める
    df['sales'] = df['sales'].fillna(sales_stats['mean'])
    
    # 「最高気温」の欠損値を平均値で埋める
    df['temperature'] = df['temperature'].fillna(temp_stats['mean'])
    
    # 「不快指数」の欠損値を平均値で埋める
    df['THI'] = df['THI'].fillna(thi_stats['mean'])
    
    # 「スタッフ数」の欠損値を平均値で埋める
    df['num_staff'] = df['num_staff'].fillna(staff_stats['mean'])
    
    # 「天気」の欠損値を「晴れ」(1)に置き換える
    df['weather'] = df['weather'].fillna(1)
    
    # 正規化関数
    def normalize(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val)
    
    # 「最高気温」の正規化
    df['temperature_norm'] = normalize(df['temperature'], temp_stats['min'], temp_stats['max'])
    
    # 「不快指数」の正規化
    df['THI_norm'] = normalize(df['THI'], thi_stats['min'], thi_stats['max'])
    
    # 「スタッフ数」の正規化
    df['num_staff_norm'] = normalize(df['num_staff'], staff_stats['min'], staff_stats['max'])
    
    # 日付でソートする
    df = df.sort_values('date')
    
    # 結果を出力する（カンマ区切り形式）
    for _, row in df.iterrows():
        print(f"{row['date']},{row['sales']},{int(row['weather'])},{row['temperature_norm']},{row['THI_norm']},{row['num_staff_norm']}")
    
    return df

def visualize_results(learn_file_path, predict_file_path, y_pred):
    """予測結果をグラフで可視化する関数"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # データの読み込み
    df_learn = pd.read_csv(learn_file_path)
    df_predict = pd.read_csv(predict_file_path)
    
    # 日付をdatetime型に変換
    df_learn['date'] = pd.to_datetime(df_learn['date'])
    df_predict['date'] = pd.to_datetime(df_predict['date'])
    
    # 予測結果をデータフレームに追加
    df_predict['predicted_sales'] = y_pred
    
    # プロットの設定
    plt.figure(figsize=(15, 8))
    
    # 学習データの売上をプロット
    plt.scatter(df_learn['date'], df_learn['sales'], 
                color='blue', alpha=0.6, label='Actual Sales (Training Data)')
    
    # 学習データの平均をプロット
    mean_sales = df_learn['sales'].mean()
    plt.axhline(y=mean_sales, color='blue', linestyle='--', 
                alpha=0.3, label=f'Mean Sales: {mean_sales:.2f}')
    
    # 予測データの売上予測をプロット
    plt.scatter(df_predict['date'], df_predict['predicted_sales'], 
                color='red', marker='x', s=100, label='Predicted Sales')
    
    # 移動平均線を追加
    window_size = 7  # 7日間の移動平均
    if len(df_learn) >= window_size:
        df_learn['sales_rolling_avg'] = df_learn['sales'].rolling(window=window_size).mean()
        plt.plot(df_learn['date'], df_learn['sales_rolling_avg'], 
                color='green', linestyle='-', label=f'{window_size}-Day Moving Average')
    
    # グラフの設定
    plt.title('Comparison of Actual and Predicted Sales', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # x軸の日付フォーマットを設定
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))  # 14日おきに表示
    plt.gcf().autofmt_xdate()  # 日付ラベルを斜めに表示
    
    # y軸のフォーマットを設定 (カンマ区切り)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # グラフの保存
    plt.tight_layout()
    plt.savefig('sales_prediction_results.png')
    print("グラフが 'sales_prediction_results.png' として保存されました。")
    
    # 追加のグラフ: 気象条件と売上の関係
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 天気と売上の関係
    weather_groups = df_learn.groupby('weather')['sales'].mean().reset_index()
    weather_labels = ['Clear', 'Sunny', 'Cloudy', 'Rainy']
    axes[0, 0].bar(weather_labels, weather_groups['sales'], color=['gold', 'skyblue', 'silver', 'lightblue'])
    axes[0, 0].set_title('Average Sales by Weather')
    axes[0, 0].set_ylabel('Average Sales')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # 2. 気温と売上の散布図
    axes[0, 1].scatter(df_learn['temperature'], df_learn['sales'], alpha=0.6)
    axes[0, 1].set_title('Temperature vs Sales')
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # 3. 不快指数と売上の散布図
    axes[1, 0].scatter(df_learn['THI'], df_learn['sales'], alpha=0.6, color='orange')
    axes[1, 0].set_title('THI (Temperature-Humidity Index) vs Sales')
    axes[1, 0].set_xlabel('THI')
    axes[1, 0].set_ylabel('Sales')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # 4. スタッフ数と売上の散布図
    axes[1, 1].scatter(df_learn['num_staff'], df_learn['sales'], alpha=0.6, color='green')
    axes[1, 1].set_title('Staff Count vs Sales')
    axes[1, 1].set_xlabel('Number of Staff')
    axes[1, 1].set_ylabel('Sales')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # グラフの保存
    plt.tight_layout()
    plt.savefig('sales_factors_analysis.png')
    print("分析グラフが 'sales_factors_analysis.png' として保存されました。")
    
    # フォント問題に対処するためのオプション設定
    # システムに適したフォントを設定する場合はコメントを外す
    # plt.rcParams['font.family'] = 'DejaVu Sans'  # または 'Arial', 'Helvetica', 'Tahoma' など
    
    # グラフを表示 (GUI環境の場合)
    try:
        plt.show()
    except:
        pass
    
    return

def step3(learn_file_path, predict_file_path):
    """Step 3: 最小二乗法を用いた売上の予測値を求める関数"""
    try:
        # デバッグ情報を格納するリスト
        debug_info = []
        
        # Step1とStep2の関数をインポート
        from weather_sales_analyzer import step1, step2
        
        # Step1で統計量を取得
        # 出力をキャプチャして表示しないようにする
        import io
        import sys as _sys
        original_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        
        # 学習データの統計量を取得
        learn_sales_stats, learn_temp_stats, learn_thi_stats, learn_staff_stats = step1(learn_file_path)
        
        # 予測データの統計量を取得
        predict_sales_stats, predict_temp_stats, predict_thi_stats, predict_staff_stats = step1(predict_file_path)
        
        _sys.stdout = original_stdout
        
        # Step2で正規化データを取得
        # 学習データの正規化
        _sys.stdout = io.StringIO()
        df_learn_normalized = step2(learn_file_path, learn_sales_stats, learn_temp_stats, learn_thi_stats, learn_staff_stats)
        _sys.stdout = original_stdout
        
        # 予測データの正規化
        _sys.stdout = io.StringIO()
        df_predict_normalized = step2(predict_file_path, predict_sales_stats, predict_temp_stats, predict_thi_stats, predict_staff_stats)
        _sys.stdout = original_stdout
        
        # 学習データと予測データを読み込む（元のデータ）
        df_learn = read_csv_file(learn_file_path)
        df_predict = read_csv_file(predict_file_path)
        
        # デバッグ情報1: データサイズ
        debug_info.append(f"DEBUG1: 学習データ行数={len(df_learn)}, 予測データ行数={len(df_predict)}")
        
        # 欠損値処理
        # 学習データの欠損値処理
        df_learn['weather'] = df_learn['weather'].fillna(1)  # 晴れ(1)で補完
        
        # 予測データの欠損値処理
        df_predict['weather'] = df_predict['weather'].fillna(1)  # 晴れ(1)で補完
        
        # 天気を4つの二値変数に変換する関数
        def weather_to_binary(weather_value):
            """天気を4つの二値変数に変換する"""
            # 0: 快晴, 1: 晴れ, 2: 曇り, 3: 雨
            is_clear = 1 if weather_value == 0 else 0
            is_sunny = 1 if weather_value == 1 else 0
            is_cloudy = 1 if weather_value == 2 else 0
            is_rainy = 1 if weather_value == 3 else 0
            return is_clear, is_sunny, is_cloudy, is_rainy
        
        # 学習データの特徴量行列（A）とターゲットベクトル（y）を作成
        A = []  # 特徴量行列
        y = []  # ターゲットベクトル（売上）
        
        for i, row in df_learn.iterrows():
            # 天気はオリジナルのデータから取得
            weather_value = row['weather']
            is_clear, is_sunny, is_cloudy, is_rainy = weather_to_binary(weather_value)
            
            # 正規化されたデータから気温、不快指数、スタッフ数と売上を取得
            normalized_row = df_learn_normalized.loc[i]
            temp_norm = normalized_row['temperature_norm']
            thi_norm = normalized_row['THI_norm']
            staff_norm = normalized_row['num_staff_norm']
            
            # 特徴ベクトル: [快晴か, 晴れか, 曇りか, 雨か, 最高気温(正規化), 不快指数(正規化), スタッフ数(正規化)]
            features = [is_clear, is_sunny, is_cloudy, is_rainy, temp_norm, thi_norm, staff_norm]
            
            A.append(features)
            # 元のデータではなく、正規化されたデータフレームから売上を取得
            y.append(normalized_row['sales'])
        
        # NumPy配列に変換
        A = np.array(A, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # デバッグ情報2: 特徴量行列の形状と売上の統計
        debug_info.append(f"DEBUG2: 特徴量行列の形状={A.shape}, 売上平均={np.mean(y):.2f}")
        
        # 最小二乗法の計算: α = (A^T A)^(-1) A^T y
        AT = A.T
        ATA = np.dot(AT, A)
        ATA_inv = np.linalg.inv(ATA)
        ATy = np.dot(AT, y)
        alpha = np.dot(ATA_inv, ATy)
        
        # デバッグ情報3: 係数
        debug_info.append(f"DEBUG3: 学習結果の係数(α)={alpha}")
        
        # 予測データの特徴量行列を作成
        A_pred = []
        
        for i, row in df_predict.iterrows():
            # 天気はオリジナルのデータから取得
            weather_value = row['weather']
            is_clear, is_sunny, is_cloudy, is_rainy = weather_to_binary(weather_value)
            
            # 正規化されたデータから気温、不快指数、スタッフ数を取得
            normalized_row = df_predict_normalized.loc[i]
            temp_norm = normalized_row['temperature_norm']
            thi_norm = normalized_row['THI_norm']
            staff_norm = normalized_row['num_staff_norm']
            
            # 特徴ベクトル: [快晴か, 晴れか, 曇りか, 雨か, 最高気温(正規化), 不快指数(正規化), スタッフ数(正規化)]
            features = [is_clear, is_sunny, is_cloudy, is_rainy, temp_norm, thi_norm, staff_norm]
            
            A_pred.append(features)
        
        # NumPy配列に変換
        A_pred = np.array(A_pred, dtype=np.float64)
        
        # デバッグ情報4: 予測用特徴量行列の形状
        debug_info.append(f"DEBUG4: 予測用特徴量行列の形状={A_pred.shape}")
        
        # 予測値の計算: y_pred = A_pred・α
        y_pred = np.dot(A_pred, alpha)
        
        # デバッグ情報5: 予測結果の統計
        debug_info.append(f"DEBUG5: 予測結果の統計: 平均={np.mean(y_pred):.2f}, 最小={np.min(y_pred):.2f}, 最大={np.max(y_pred):.2f}")
        
        # デバッグ情報6: 予測データのサンプル
        sample_size = min(3, len(y_pred))
        sample_indices = np.round(np.linspace(0, len(y_pred) - 1, sample_size)).astype(int)
        samples = [f"y_pred[{i}]={y_pred[i]:.2f}" for i in sample_indices]
        debug_info.append(f"DEBUG6: サンプル予測値: {', '.join(samples)}")
        
        # デバッグ情報を出力（6行以内）
        for line in debug_info:
            print(line)
        
        # 結果を出力（1行に1つの予測値）
        for pred in y_pred:
            print(f"{pred}")
        
        visualize_results(learn_file_path, predict_file_path, y_pred)

        
        return y_pred
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python main.py [path_to_csv] またはステップ名と必要なパラメータ")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "step1":
        if len(sys.argv) < 3:
            print("使用方法: python main.py step1 [path_to_csv]")
            sys.exit(1)
        
        file_path = sys.argv[2]
        step1(file_path)
        
    elif command == "step2":
        if len(sys.argv) < 3:
            print("使用方法: python main.py step2 [path_to_csv]")
            sys.exit(1)
        
        file_path = sys.argv[2]
        # まずstep1を実行して統計量を取得（結果の出力をキャプチャして表示しないようにする）
        import io
        import sys as _sys
        original_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        sales_stats, temp_stats, thi_stats, staff_stats = step1(file_path)
        _sys.stdout = original_stdout
        
        # step2を実行
        step2(file_path, sales_stats, temp_stats, thi_stats, staff_stats)
        
    elif command == "step3":
        if len(sys.argv) < 4:
            print("使用方法: python main.py step3 [path_to_learn_csv] [path_to_predict_csv]")
            sys.exit(1)
        
        learn_file_path = sys.argv[2]
        predict_file_path = sys.argv[3]
        step3(learn_file_path, predict_file_path)
        
    else:
        # 単一のCSVファイルが与えられた場合
        file_path = sys.argv[1]
        sales_stats, temp_stats, thi_stats, staff_stats = step1(file_path)
        step2(file_path, sales_stats, temp_stats, thi_stats, staff_stats)

if __name__ == "__main__":
    main()