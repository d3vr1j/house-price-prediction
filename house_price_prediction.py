"""
House Price Prediction using Linear Regression
Dataset: Kaggle - House Prices: Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Predicts SalePrice from:
    - Square footage (GrLivArea: above-ground living area in sq ft)
    - Number of bedrooms (BedroomAbvGr)
    - Number of bathrooms (full + half baths, above ground and basement)
"""

import argparse
import os
import urllib.request  # ADDED: to auto-download the dataset

import matplotlib
matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["SquareFootage", "Bedrooms", "Bathrooms"]
TARGET = "SalePrice"

# ADDED: public GitHub mirror of the Kaggle files, used when data/ is empty
DATA_URLS = {
    "train.csv": "https://raw.githubusercontent.com/mkbehbehani/spark-advanced-regression-kaggle/master/source-data/train.csv",
    "test.csv": "https://raw.githubusercontent.com/mkbehbehani/spark-advanced-regression-kaggle/master/source-data/test.csv",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three model features from the raw Kaggle columns."""
    out = pd.DataFrame(index=df.index)
    out["SquareFootage"] = df["GrLivArea"]
    out["Bedrooms"] = df["BedroomAbvGr"]
    # Half baths count as 0.5 of a full bath
    out["Bathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    # Fill any remaining missing values with the column median
    return out.fillna(out.median())


def download_if_missing(data_dir: str) -> None:  # ADDED
    """Download train.csv and test.csv into data_dir if they are not already there."""
    os.makedirs(data_dir, exist_ok=True)
    for name, url in DATA_URLS.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Download train.csv from Kaggle and place it in the data/ folder."
        )
    return pd.read_csv(path)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    y_pred = model.predict(X_test)
    print("\nModel Evaluation (test set)")
    print("-" * 35)
    print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE      : ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE     : ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    return y_pred


def print_coefficients(model: LinearRegression) -> None:
    print("\nModel Coefficients")
    print("-" * 35)
    print(f"Intercept      : {model.intercept_:,.2f}")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"{name:<15}: {coef:,.2f}")


def plot_results(y_test: pd.Series, y_pred: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolor="k")
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual ($)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals.png"), dpi=120)
    plt.close()

    print(f"\nPlots saved to '{out_dir}/'")


def make_submission(model: LinearRegression, test_path: str, out_path: str) -> None:
    """Predict on Kaggle's test.csv and write a submission file."""
    if not os.path.exists(test_path):
        print(f"\n'{test_path}' not found, skipping Kaggle submission.")
        return
    test_df = pd.read_csv(test_path)
    preds = model.predict(build_features(test_df))
    preds = np.clip(preds, 0, None)  # prices cannot be negative
    pd.DataFrame({"Id": test_df["Id"], "SalePrice": preds}).to_csv(out_path, index=False)
    print(f"Submission saved to '{out_path}'")


def predict_price(model: LinearRegression, sqft: float, bedrooms: int, bathrooms: float) -> float:
    sample = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=FEATURES)
    return float(model.predict(sample)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="House price prediction with linear regression")
    parser.add_argument("--data-dir", default="data", help="Folder containing train.csv and test.csv")
    parser.add_argument("--output-dir", default="outputs", help="Folder for plots and submission")
    args = parser.parse_args()

    # 1. Load data (downloads it first if data/ is empty)
    download_if_missing(args.data_dir)  # ADDED
    df = load_data(os.path.join(args.data_dir, "train.csv"))
    print(f"Loaded {len(df)} rows")

    # 2. Build features and target
    X = build_features(df)
    y = df[TARGET]
    print("\nFeature summary")
    print(X.describe().round(2))

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train
    model = train_model(X_train, y_train)
    print_coefficients(model)

    # 5. Evaluate
    y_pred = evaluate(model, X_test, y_test)

    # 6. Plots
    plot_results(y_test, y_pred, args.output_dir)

    # 7. Kaggle submission (optional, needs test.csv)
    make_submission(model, os.path.join(args.data_dir, "test.csv"),
                    os.path.join(args.output_dir, "submission.csv"))

    # 8. Example prediction
    price = predict_price(model, sqft=2000, bedrooms=3, bathrooms=2.5)
    print(f"\nExample: 2000 sq ft, 3 bed, 2.5 bath -> predicted price ${price:,.2f}")


if __name__ == "__main__":
    main()
