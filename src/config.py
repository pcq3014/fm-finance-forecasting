from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Định nghĩa các thư mục quan trọng
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UNIFIED_DATA_DIR = RAW_DATA_DIR / "unified"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

for folder in [RAW_DATA_DIR, UNIFIED_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(folder, exist_ok=True)

# 4. Các tham số mô hình (Có thể để đây hoặc đọc từ file yaml)
MODEL_PARAMS = {
    "chronos": {"batch_size": 32, "epochs": 10},
    "llama": {"context_length": 512}
}
