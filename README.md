# 🚗 Turbo.az Car Price Prediction

Predicting car prices from [Turbo.az](https://turbo.az) listings using multiple regression models, including **LightGBM**, **XGBoost**, **Ridge**, **Linear Regression**, and a **Stacking Regressor**.

---

## 📦 Dataset

- **Source:** Scraped from [Turbo.az](https://turbo.az)
- **File:** `cars.csv`
- **Size:** ~432,000 car listings
- **Target:** Car price in AZN

---

## 🧹 Data Preprocessing

- Dropped duplicate price column (`price_y` ≈ `price_x`)
- Converted USD (`$`) and EUR (`€`) prices to AZN (×1.7 and ×1.95)
- Extracted horsepower (`guc`) from the engine (`Mühərrik`) column via regex
- Encoded crash status (`spare_parts`): crashed = `1`, otherwise = `0`
- Dropped high-missing columns: `Qəzalı`, `vin`, `loan`, `featured`, `salon`, `shop_name`, `barter`, `vip`
- Removed rows with missing `Sahiblər` (number of owners)

---

## 📊 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `production_year` | Numeric | Year of manufacture |
| `engine_displacement_num` | Numeric | Engine volume (L) |
| `kilometrage_num` | Numeric | Mileage (km) |
| `guc` | Numeric | Horsepower (extracted) |
| `spare_parts` | Numeric | Crashed car flag (0/1) |
| `Sahiblər` | Numeric | Number of previous owners |
| `Marka` | Categorical | Car brand |
| `Model` | Categorical | Car model |
| `Ban növü` | Categorical | Body type |
| `Sürətlər qutusu` | Categorical | Transmission |
| `Ötürücü` | Categorical | Drive type |
| `Rəng` | Categorical | Color |
| `Vəziyyəti` | Categorical | Condition |
| `Yeni` | Categorical | New/Used |
| `Hansı bazar üçün yığılıb` | Categorical | Market (EU/local) |
| `Yerlərin sayı` | Categorical | Number of seats |
| `city` | Categorical | City |
| `extra_info` | Categorical | Extra features |

---

## ⚙️ Preprocessing Pipeline

- **Numeric:** Median imputation → Standard scaling
- **Categorical:** Most-frequent imputation → One-hot encoding
- **Train/Test split:** 80% / 20% (`random_state=42`)

---

## 🤖 Models

| Variable | Model | Test R² |
|----------|-------|---------|
| `lr_model` | Linear Regression | ~0.77 |
| `ridge_model` | Ridge Regression | ~0.77 |
| `lgbm_model` | LightGBM | ~0.97 |
| `xgb_model` | XGBoost | ~0.97 |
| `stack_model` | Stacking Regressor | ~0.97 |

> **Best model:** LightGBM / XGBoost / Stacking Regressor with Test R² ≈ 0.97

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/edhaevv/turboaz-price-prediction.git
cd turboaz-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Place `cars.csv` in the project root directory.

### 4. Run the notebook
Open `Turboaz_clean.ipynb` in Jupyter or Google Colab and run all cells.

---

## 🔍 Inference Example

```python
new_car = pd.DataFrame([{
    'production_year': 2008,
    'engine_displacement_num': 1.3,
    'kilometrage_num': 378000,
    'Ban növü': 'Hetçbek',
    'Marka': 'Opel',
    'Model': 'Astra',
    'guc': 100,
    # ... other features
}])

predicted_price = stack_model.predict(new_car)[0]
print(f"Predicted price: {predicted_price:,.0f} AZN")
```

---

## 📁 Project Structure

```
├── Turboaz_clean.ipynb    # Main notebook
├── cars.csv               # Dataset (not included)
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files
└── README.md              # This file
```

---

## 🛠️ Requirements

See `requirements.txt`. Main dependencies:

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `matplotlib`, `seaborn`
