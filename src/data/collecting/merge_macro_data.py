import pandas as pd
from pathlib import Path

def merge_macro_data():
    # Lấy đường dẫn thư mục gốc của project
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # 1. Đường dẫn tới các file dữ liệu
    macro_vn_path = project_root / 'data' / 'macro' / 'macro_vn.csv'
    pmi_path = project_root / 'data' / 'macro' / 'pmi.csv'
    ibor_path = project_root / 'data' / 'macro' / 'ibor_rate.csv'

    # 2. Đọc các file csv
    df_macro = pd.read_csv(macro_vn_path)
    df_pmi = pd.read_csv(pmi_path)
    df_ibor = pd.read_csv(ibor_path)

    # 3. Chuẩn hóa cột Date sang dạng datetime để merge chính xác
    # Chú ý: Cột Date được tự động parse sang dạng YYYY-MM-DD
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_pmi['Date'] = pd.to_datetime(df_pmi['Date'])
    df_ibor['Date'] = pd.to_datetime(df_ibor['Date'])

    # 4. Gộp dữ liệu
    # Dùng how='outer' để không bị mất các ngày không trùng khớp, hoặc 'left' nếu bạn chỉ muốn giữ các ngày có trong macro_vn
    df_merged = df_macro.merge(df_ibor, on='Date', how='outer')
    df_merged = df_merged.merge(df_pmi, on='Date', how='outer')

    # 5. Sắp xếp dữ liệu lại theo thứ tự ngày tháng
    df_merged = df_merged.sort_values(by='Date').reset_index(drop=True)

    # (Tùy chọn) Do PMI là dữ liệu công bố theo tháng, bạn có thể fill forward (ffill) 
    # cho các ngày còn lại trong tháng đó:
    # df_merged['vietnam_pmi'] = df_merged['vietnam_pmi'].ffill()

    # 6. Lưu file
    output_path = project_root / 'data' / 'macro' / 'macro_vn.csv'
    df_merged.to_csv(output_path, index=False)
    
    print(f"Đã gộp xong! Kết quả được lưu tại: {output_path}")

if __name__ == "__main__":
    merge_macro_data()