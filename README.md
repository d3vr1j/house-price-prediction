# House Price Prediction (Linear Regression)

A linear regression model that predicts house prices from square footage, number of bedrooms and number of bathrooms.

Dataset: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## Features Used

| Feature | Source column(s) |
|---|---|
| Square footage | `GrLivArea` (above-ground living area) |
| Bedrooms | `BedroomAbvGr` |
| Bathrooms | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` |

Target: `SalePrice`

## Project Structure

```
house-price-prediction/
├── data/                        # train.csv and test.csv (auto-downloaded, not committed)
├── outputs/                     # plots and submission.csv (generated)
├── house_price_prediction.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

The dataset is downloaded automatically into `data/` on the first run, so no Kaggle login is needed. You can also download `train.csv` and `test.csv` from Kaggle yourself and place them in `data/`.

## Run

```bash
python house_price_prediction.py
```

Optional arguments:

```bash
python house_price_prediction.py --data-dir data --output-dir outputs
```

## Output

- Model coefficients and intercept
- Evaluation metrics on a 20% hold-out set: R2, MAE, RMSE
- `outputs/actual_vs_predicted.png` and `outputs/residuals.png`
- `outputs/submission.csv` for Kaggle (if `test.csv` is present)
- An example prediction for a 2000 sq ft, 3 bed, 2.5 bath house

## Results

| Metric | Value |
|---|---|
| R2 | 0.658 |
| MAE | $34,396 |
| RMSE | $51,223 |

Each extra square foot adds about $95 to the predicted price, and each extra bathroom adds about $30,000. The bedroom coefficient is negative: for a house of the same size, more bedrooms means smaller rooms.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib
