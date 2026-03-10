# 📈 Airmiles Time Series Forecasting with Holt-Winters

> Forecast the next 12 months of airline passenger miles using **Holt-Winters (Triple Exponential Smoothing)**.

---

## 🎯 Objective

This project fulfills the **Holt-Winters Challenge** by:

- Setting the time series frequency to `"MS"` (Month Start).
- Using the **last 12 months as the test set**.
- Training a **Holt-Winters model** only on the training period.
- Forecasting **12 months ahead**.
- Evaluating model accuracy using **Mean Absolute Error (MAE)**.

## 📊 Workflow Summary

1. **Data Frequency**  
   - Loaded `airmiles.csv` with `Date` as index.  
   - Set frequency explicitly to `"MS"` (Month Start), as required.

2. **Visualization**  
   - Plotted the raw time series to observe trend and variability.  
   - Used `month_plot()` and `quarter_plot()` to explore seasonal patterns.  
   - Applied multiplicative seasonal decomposition (`period=12`) to isolate:  
     - Level  
     - Trend  
     - Seasonality  
     - Residuals

3. **Train/Test Split**  
   - **Training set**: all data except the last 12 months.  
   - **Test set**: last 12 months (unseen during training).

4. **Modeling**  
   - Used `ExponentialSmoothing` with:  
     - `trend='mul'`  
     - `seasonal='mul'`  
     - `seasonal_periods=12`  
   - Fitted only on the training data.

5. **Forecasting**  
   - Generated 12-step-ahead forecast.  
   - Visualized training data, test data, and predictions on the same plot.

6. **Accuracy Assessment**  
   - Computed **MAE** (primary metric per challenge).  
   - Also reported **RMSE** and **MAPE** for context.  

7. **Final Forecast**  
- Retrained the model on the full dataset.  
- Produced a 12-month forecast into the future.  
- Plotted historical + forecasted values.

  
## 📝 Notes
- This approach provides a transparent, interpretable baseline for time series forecasting.
- The multiplicative model was chosen because seasonal fluctuations grow over time (confirmed via decomposition).
- All steps follow best practices for unbiased model evaluation.


