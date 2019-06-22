import argparse
import traceback
import sys
import os
import os.path

import xgboost as xgb
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply XGBoost regression model")

    parser.add_argument("--model", type=str, metavar="arg", required=True,
                        help="saved XGBoost model")
    parser.add_argument("--data", type=str, metavar="arg", required=True,
                        help="data in CSV format")
    parser.add_argument("--result", type=str, metavar="arg", required=True,
                        help="file where to write predictions")

    args = parser.parse_args()

    return args


def main():
    try:
        args = parse_args()

        df = pd.read_csv(args.data, dtype=np.float32)

        regressor = xgb.XGBRegressor()
        regressor.load_model(args.model)

        preds = regressor.predict(df)

        result = pd.DataFrame({"prediction": preds})
        result.to_csv(args.result, float_format="%.3f", index=False, header=False)
    except Exception as exc:
        print(exc, file=sys.stderr)
        print("===", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(-1)


if __name__ == "__main__":
    main()
