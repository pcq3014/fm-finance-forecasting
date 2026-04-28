import requests
import pandas as pd
import time
from datetime import datetime
import os

def get_data_vndirect_unified(symbol, start_date, end_date):
    """
    Hàm lấy dữ liệu TẤT CẢ (Cổ phiếu & Chỉ số) từ VNDirect
    """
    # 1. Setup thời gian
    start_ts = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_ts = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    
    # 2. API Endpoint
    url = "https://dchart-api.vndirect.com.vn/dchart/history"
    params = {
        'resolution': 'D',
        'symbol': symbol,
        'from': start_ts,
        'to': end_ts
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    print(f"[{symbol}] Đang tải dữ liệu...", end=" ")
    
    try:
        response = requests.get(url, params=params, headers=headers)
        
        # Kiểm tra lỗi server
        if response.status_code != 200:
            print(f"Lỗi HTTP {response.status_code}")
            return None
            
        data = response.json()
        
        if 't' not in data or len(data['t']) == 0:
            print("-> Không có dữ liệu.")
            return None
            
        # 3. Tạo DataFrame
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['t'], unit='s'),
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'], # VNDirect mặc định là giá ĐÃ ĐIỀU CHỈNH
            'Volume': data['v']
        })
        
        # 4. Xử lý múi giờ
        df.set_index('Date', inplace=True)
        df.index = df.index + pd.Timedelta(hours=7)
        df.index = df.index.normalize()
        
        print(f"-> Thành công! ({len(df)} dòng)")
        return df

    except Exception as e:
        print(f"-> Lỗi: {e}")
        return None

# --- CHẠY VÒNG LẶP LẤY TOÀN BỘ ---

# Danh sách mã cần lấy (Bao gồm cả Cổ phiếu và Index)
tickers = [
    "VNINDEX", "VN30",      # Chỉ số
    "VCB", "FPT", "HPG",    # Cổ phiếu
    "SSI", "VNM", "SSI", "VHM"
]

# Tạo thư mục
save_dir = "data/raw/unified"
os.makedirs(save_dir, exist_ok=True)

for ticker in tickers:
    # Gọi hàm
    df = get_data_vndirect_unified(ticker, "2013-01-01", "2025-12-31")
    
    if df is not None:
        # Lưu file
        df.to_csv(f"{save_dir}/{ticker}.csv")
    
    # QUAN TRỌNG: Nghỉ 2 giây để tránh bị VNDirect chặn IP
    time.sleep(2) 

print("\nĐã hoàn thành tải toàn bộ dữ liệu!")