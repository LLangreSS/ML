# 💰 Daily Revenue Forecasting with SARIMAX

> Forecast future daily revenue using **Seasonal ARIMA with eXogenous regressors (SARIMAX)**, incorporating promotional activity such as discounts and coupons.

---

## 🎯 Objective

This project demonstrates a rigorous, production-grade approach to time series forecasting by:

- Modeling **daily revenue** as a seasonal time series with weekly patterns.
- Incorporating **exogenous variables**: `discount_rate` and `coupon_rate`.
- Using **time series cross-validation** (forward-chaining) for robust model selection.
- Tuning SARIMAX hyperparameters via grid search.
- Producing **out-of-sample forecasts** aligned with future marketing plans.

## 📊 Workflow Summary

1. **Data Preparation**  
   - Loaded `daily_revenue.csv` with date as index and daily frequency (`"D"`).  
   - Cleaned numeric fields: removed commas from `revenue`, stripped `%` from rates.  
   - Ensured consistent scaling: historical rates as percentages, future regressors as proportions (×100 applied internally).

2. **Exploratory Analysis**  
   - Plotted raw revenue to inspect trend, volatility, and outliers.  
   - Used `month_plot()` and `quarter_plot()` to detect seasonal patterns.  
   - Performed **multiplicative seasonal decomposition** (`period=365`) to separate trend, seasonality, and residuals.  
   - Analyzed **ACF/PACF** to inform AR/MA order selection.

3. **Stationarity Assessment**  
   - Conducted **Augmented Dickey–Fuller (ADF) test**: original series non-stationary.  
   - Applied **first-order differencing** (`d=1`), confirmed stationarity (p ≈ 0).

4. **Model Progression**  
   - Started with **ARIMA(p,d,q)**.  
   - Extended to **SARIMA** with weekly seasonality (`s=7`).  
   - Final model: **SARIMAX** with `exog = [discount_rate, coupon_rate]`.

5. **Validation Strategy**  
   - Used **`TimeSeriesSplit`** with 5 folds and 30-day test horizons.  
   - Avoided random splits — preserved temporal order to prevent data leakage.  
   - Evaluated models using **MAE**, **RMSE**, and **MAPE**.

6. **Hyperparameter Tuning**  
   - Grid-searched over `(p,d,q)` and seasonal `(P,D,Q)` orders.  
   - Selected best configuration based on **average RMSE** across CV folds.  
   - Saved optimal parameters to `best_params_sarimax.csv`.

7. **Final Forecast**  
   - Refit the best SARIMAX model on **full historical data**.  
   - Forecasted revenue for future dates using planned `discount_rate` and `coupon_rate` from `future_regressors.csv`.  
   - Visualized historical trend + forecast in a single plot.

## 📝 Notes

- The model assumes **no missing values** and **strict daily frequency** — prerequisites for reliable state-space estimation.
- `simple_differencing=False` ensures forecasts are on the **original revenue scale**, not differenced.
- Exogenous regressors are estimated via **MLE** (`mle_regression=True`), yielding stable coefficient estimates.
- This pipeline respects the **causal structure** of time series: no future information leaks into training.
- The approach is **interpretable**, **reproducible**, and ready for extension (e.g., holiday dummies, event indicators).
