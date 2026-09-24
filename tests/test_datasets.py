import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.data.loaders import load_dataset

for d in ["german_credit", "taiwan_credit", "adult_income"]:
    try:
        print(f"Testing {d}...")
        res = load_dataset(d)
        print(f"  {d} SUCCESS: X shape {res['X'].shape}, y shape {res['y'].shape}, groups {set(res['sensitive'])}")
    except Exception as e:
        print(f"  {d} ERROR: {e}")
