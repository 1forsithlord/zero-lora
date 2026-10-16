import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from zerolora.frozen_eval import main

if __name__ == "__main__":
    main()
