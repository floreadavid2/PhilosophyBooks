from pathlib import Path
import sys
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=True)