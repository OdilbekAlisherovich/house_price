#  House Prices

Predict house sale prices using the Ames Housing dataset.  
**Metric:** RMSE on log-transformed SalePrice (lower is better)

## Results

| Model | CV Log RMSE |
XGBoost  RMSE: 0.11466 ± 0.00522
LightGBM RMSE: 0.12075 ± 0.00851

## What was done

**Preprocessing**
- Removed 2 outliers based on `GrLivArea`
- Filled missing values with `None`, `0`, or neighborhood median

**Feature Engineering**
- `TotalSF` — total house area
- `TotalBath` — total bathrooms
- `HouseAge`, `RemodAge`, `GarageAge` — age-based features
- `QualSF`, `QualGrLiv` — quality × area interactions
- Binary flags: `HasPool`, `HasGarage`, `HasBsmt`, etc.

**Modeling**
- XGBoost and LightGBM tuned with `GridSearchCV`
- Combined using `StackingRegressor` with Ridge as meta-learner

## How to run

```bash
pip install -r requirements.txt
jupyter notebook house_price_model.ipynb
```
## Project structure

```
house-prices/
├── house_price.ipynb
├── README.md
└── requirements.txt
```