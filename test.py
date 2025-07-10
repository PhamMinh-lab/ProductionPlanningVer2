import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pulp
import base64
from io import BytesIO

# ========== Sidebar Navigation ==========
st.sidebar.title("⚙️ Settings")
section = st.sidebar.selectbox("Select Section", ["Project Overview", "Forecasting & Planning", "Project Process"])

# ========== Load and Preprocess Data ==========
@st.cache_data
def load_data():
    df = pd.read_excel("B202.xlsx", sheet_name='Demand')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Sản lượng']].set_index('Date').asfreq('MS')
    df = df.interpolate(method='linear')
    return df

data = load_data()

@st.cache_data
def preprocess_data(data):
    def remove_outliers(df, col_name, lower=0.25, upper=0.75, threshold=1.2):
        Q1 = df[col_name].quantile(lower)
        Q3 = df[col_name].quantile(upper)
        IQR = Q3 - Q1
        upper_bound = Q3 + threshold * IQR
        lower_bound = Q1 - threshold * IQR
        return df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]

    df = remove_outliers(data.copy(), 'Sản lượng')
    df['lag_1'] = df['Sản lượng'].shift(1)
    df['lag_2'] = df['Sản lượng'].shift(2)
    df['lag_12'] = df['Sản lượng'].shift(12)
    df['rolling_mean_3'] = df['Sản lượng'].rolling(window=3).mean().shift(1)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    rain_months = [f"{y}-{str(m).zfill(2)}" for y in range(2012, 2025) for m in range(6, 10)]
    tet_months = ['2012-02', '2013-03', '2014-02', '2015-03', '2016-03', '2017-02',
                  '2018-03', '2019-03', '2020-02', '2021-02', '2022-03', '2023-03', '2024-01','2025-02']

    df['Rain'] = df.index.strftime('%Y-%m').isin(rain_months).astype(int)
    df['Tet'] = df.index.strftime('%Y-%m').isin(tet_months).astype(int)
    df = df.dropna()

    features = ['Rain', 'Tet', 'lag_1', 'lag_2', 'lag_12', 'rolling_mean_3', 'month', 'quarter']
    num_features = ['Sản lượng', 'lag_1', 'lag_2', 'lag_12', 'rolling_mean_3']

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[num_features]),
                             index=df.index,
                             columns=num_features)
    df_scaled = pd.concat([df_scaled, df[features[0:2] + ['month', 'quarter']]], axis=1)
    return df, df_scaled, scaler, features

demand, demand_scaled, scaler, features = preprocess_data(data)

# ========== Project Process ==========
if section == "Project Process":
    st.markdown(
    "<h1 style='text-align: center;'>🛠️ Project Process",
    unsafe_allow_html=True)
    st.markdown("""
    #### Step 1: Load Data
    - Read Excel file with monthly demand.

    #### Step 2: Data Cleaning
    - Interpolate missing values.
    - Remove outliers using IQR method.

    #### Step 3: Feature Engineering
    - Lag features, rolling mean, month/quarter, rain/Tet events.

    #### Step 4: Train Models
    - **Statistical Models**: ARIMA, SARIMA, SARIMAX, Exponential Smoothing, Prophet
    - **Machine Learning**: Random Forest, SVM, XGBoost, Linear, Polynomial Regression

    #### Step 5: Evaluation
    - MAE, MSE, R² Metrics
    - **Random Forest is chosen** due to lowest MAE/MSE and highest R².
    """)

    results_df = pd.read_excel("results.xlsx", index_col=0)
    st.subheader("📊 Model Performance Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best MAE", f"{results_df['MAE'].min():,.2f}")
    with col2:
        st.metric("Best MSE", f"{results_df['MSE'].min():,.2f}")
    with col3:
        st.metric("Best R²", f"{results_df['R2'].max():.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(18, 6))
    axes[0].set_title("MAE")
    axes[1].set_title("MSE")
    axes[2].set_title("R²")

    st.subheader("MSE for Models (lower is better)")
    colors = ['red' if idx == 'Random Forest' else '#1f77b4' for idx in results_df.index]
    st.bar_chart(results_df[['MSE']].assign(color=colors).drop(columns='color'))

    st.subheader("MAE for Models (lower is better)")
    colors = ['red' if idx == 'Random Forest' else '#1f77b4' for idx in results_df.index]
    st.bar_chart(results_df[['MAE']].assign(color=colors).drop(columns='color'))

    st.subheader("R2 for Models (Higher is better)")
    colors = ['red' if idx == 'Random Forest' else '#1f77b4' for idx in results_df.index]
    st.bar_chart(results_df[['R2']].assign(color=colors).drop(columns='color'))

    st.subheader("Random Forest is chosen due to lowest MAE/MSE and highest R²")
# ========== Forecasting & Planning ==========
if section == "Forecasting & Planning":
    st.markdown(
    "<h1 style='text-align: center;'>📊 Demand Forecasting & Production Planning</h1>",
    unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    current_date_str = col1.text_input("Current Date (YYYY-MM)", "2025-07")
    inventory = col2.number_input("Current Inventory", min_value=0, value=10000)
    n_months = col3.slider("Forecast Horizon (Months)", 1, 6, 3)

    try:
        current_date = pd.to_datetime(current_date_str + "-01")
        start_date = current_date + pd.DateOffset(months=1)
        future_dates = pd.date_range(start=start_date, periods=n_months, freq='MS')

        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        X = demand_scaled[features]
        y = demand_scaled[['Sản lượng']]
        model.fit(X, y)

        forecast = []
        history = demand_scaled[demand_scaled.index < start_date].copy()

        for date in future_dates:
            lag_1 = history.iloc[-1]['Sản lượng']
            lag_2 = history.iloc[-2]['Sản lượng']
            lag_12 = history.iloc[-12]['Sản lượng'] if len(history) >= 12 else lag_1
            rolling_mean_3 = history['Sản lượng'].iloc[-3:].mean()
            month = date.month
            quarter = (month - 1) // 3 + 1
            rain = int(date.strftime('%Y-%m') in history.index.strftime('%Y-%m'))
            tet = int(date.strftime('%Y-%m') in ['2025-02'])

            X_future = pd.DataFrame.from_dict([{
                'Rain': rain, 'Tet': tet, 'lag_1': lag_1,
                'lag_2': lag_2, 'lag_12': lag_12, 'rolling_mean_3': rolling_mean_3,
                'month': month, 'quarter': quarter
            }])

            y_pred_scaled = model.predict(X_future)[0]
            input_scaled = np.array([[y_pred_scaled, lag_1, lag_2, lag_12, rolling_mean_3]])
            y_pred_real = scaler.inverse_transform(input_scaled)[0, 0]

            forecast.append((date.strftime('%Y-%m'), y_pred_real))

            history.loc[date] = {
                'Sản lượng': y_pred_scaled,
                'lag_1': lag_1, 'lag_2': lag_2,
                'lag_12': lag_12, 'rolling_mean_3': rolling_mean_3,
                'Rain': rain, 'Tet': tet, 'month': month, 'quarter': quarter
            }

        forecast_df = pd.DataFrame(forecast, columns=['Month', 'Forecasted Demand'])
        st.subheader("🔮 Forecasted Demand")
        st.dataframe(forecast_df)

        # === PRODUCTION PLANNING ===
        months = forecast_df['Month'].tolist()
        demand_values = forecast_df['Forecasted Demand'].tolist()
        capacity = [150000 if pd.to_datetime(m).month in [5,6,7,8,9,10] else 200000 for m in months]

        model_lp = pulp.LpProblem("Production_Plan", pulp.LpMinimize)
        Prod = pulp.LpVariable.dicts("Prod", months, lowBound=0, cat='Continuous')
        Inv = pulp.LpVariable.dicts("Inv", months, lowBound=0, cat='Continuous')

        model_lp += pulp.lpSum(Prod[m] for m in months)

        for i, m in enumerate(months):
            if i == 0:
                model_lp += Inv[m] == inventory + Prod[m] - demand_values[i]
            else:
                model_lp += Inv[m] == Inv[months[i-1]] + Prod[m] - demand_values[i]
            model_lp += Prod[m] <= capacity[i]
            model_lp += Inv[m] >= 5000
            model_lp += Inv[m] <= 5000

        model_lp.solve()

        plan = [(m, Prod[m].varValue, Inv[m].varValue) for m in months]
        plan_df = pd.DataFrame(plan, columns=['Month', 'Production', 'Inventory'])
        st.subheader("🏭 Production Plan")
        st.dataframe(plan_df)


        # === Plot ===
        st.line_chart(plan_df.set_index('Month')[['Production', 'Inventory']])

        # === Combined Output & Download ===
        combined_df = forecast_df.merge(plan_df, on='Month')
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        st.download_button("⬇️ Download", convert_df(combined_df), file_name="forecast_production_plan.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ========== Project Overview ==========
if section == "Project Overview":
    st.markdown(
    "<h1 style='text-align: center;'>📊 Project Overview",
    unsafe_allow_html=True)
    st.subheader("Context:")
    st.text("In the era of automation in manufacturing, effective production planning plays a crucial role in the efficiency of a business. In the traditional planning tasks, there are many problems such as inaccurate demand forecasting and poor production schedule, which leads to various negative consequences and affects to all parts of the supply chain.")
    st.subheader("Objective:")
    st.text("This project aims to integrate machine learning-based demand forecasting with optimization models to generate more accurate and efficient monthly production plans.")  
    st.subheader("Scenario:")
    st.text("The inadequacy of the current forecasting methodology is clearly reflected in the data collected for the year 2024. Monthly comparisons between actual sales volume and production output reveal significant discrepancies, particularly during periods of heightened demand. For example, in January, the sales volume reached 123,404 units, whereas only 109,217 units were produced, resulting in a supply shortfall of over 14,000 units. Similarly, in December, sales volume reached 101,386 units, exceeding production output of 85,140 units by more than 16,000 units. Such patterns are not isolated cases but recur throughout the year, especially in months such as March, November, and December. These observations point to a fundamental weakness in the company’s ability to accurately forecast demand and align production capacity accordingly. While production may occasionally exceed sales in certain months such as April and May, the overall mismatch indicates a reactive production strategy rather than a proactive and forecast-driven one.")
    compare_production_sales = pd.read_excel("Book2.xlsx")
    st.area_chart(compare_production_sales, x='Month', y=['Production','Demand'],color=["#0000FF","#FF0000"], stack=False)
    st.text("It is also important to emphasize that the existing forecasting methodology—based largely on historical sales and current orders—may provide some utility in the short term. However, it fails to deliver meaningful insights for medium- and long-term planning. In a competitive market environment characterized by fluctuating customer preferences, seasonal variability, and economic uncertainty, reliance on past data without the use of formal forecasting techniques such as time series analysis or regression modeling severely limits the company planning capabilities.")
