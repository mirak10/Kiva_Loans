import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Step 1: Data loading and exploring
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


# Step 2: Data Cleaning and transformation
def clean_data(df):
    df['partner_id'] = df['partner_id'].fillna(-1)  # replace the na values with -1 to indicate not assigned
    df['borrower_genders'] = df['borrower_genders'].fillna('unknown')  # dropped na values of borrower genders
    df['date'] = pd.to_datetime(df['date'])  # change the date column to datetime format
    df['year'] = df['date'].dt.year  # Creating new features i.e. year & month. Will be useful in time analysis
    df['month'] = df['date'].dt.month
    return df


def count_gender(genders):
    if pd.isna(genders) or genders.strip() == "":
        return pd.Series([0, 0, 1])
    genders = genders.lower().split()
    female = genders.count('female')
    male = genders.count('male')
    unknown = 1 if female == 0 and male == 0 else 0
    return pd.Series([female, male, unknown])


def apply_gender_split(df):
    df[['female_count', 'male_count', 'unknown_count']] = df['borrower_genders'].apply(count_gender)
    return df


# Data visualization
# Scatterplot: Loan Amount vs Funded Amount

# def visualize_loan_vs_funding(df):
#     sns.scatterplot(x='loan_amount', y='funded_amount', data=df)
#     plt.title('Loan Amount vs Funded Amount')
#     plt.xlabel('Requested')
#     plt.ylabel('Funded')
#     plt.show()

# Step 3: Machine learning. For predicting funded amount

def prepare_features(df):
    features = ['loan_amount', 'sector', 'country', 'term_in_months', 'lender_count',
                'female_count', 'male_count', 'unknown_count']
    target = 'funded_amount'
    X = df[features]
    y = df[target]
    return X, y


def build_preprocessor():
    # Define categorical and numerical columns
    categorical_cols = ['sector', 'country']

    # One-hot encoder for categorical, passthrough for numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'  # Keeps the rest (numerical)
    )
    return preprocessor


def build_model_pipeline(preprocessor):
    # Random Forest regressor pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return model


def train_model(X, y, model):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # manually calculate RMSE
    r2 = r2_score(y_test, y_pred)

    # Visualize predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
    plt.xlabel("Actual Funded Amount")
    plt.ylabel("Predicted Funded Amount")
    plt.title("Actual vs Predicted Funded Amounts")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, X_test, y_test, y_pred


def run_funded_amount_prediction(df):
    X, y = prepare_features(df)
    preprocessor = build_preprocessor()
    model = build_model_pipeline(preprocessor)
    trained_model, X_test, y_test, y_pred = train_model(X, y, model)
    return trained_model


def prepare_time_series(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df


def plot_loan_amount_over_time(df):
    plt.figure(figsize=(15, 5))
    df['loan_amount'].plot()
    plt.title("Loan Amount Over Time")
    plt.xlabel("Date")
    plt.ylabel("Loan Amount")
    plt.grid(True)
    plt.show()


def forecast_loan_amount_arima(df):
    ts = df['loan_amount'].resample('Y').sum()

    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())

    forecast = model_fit.forecast(steps=5)
    forecast.index = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1), periods=5, freq='Y')

    plt.figure(figsize=(12, 5))
    ts.plot(label='Actual')
    forecast.plot(label='Forecast', color='red')
    plt.legend()
    plt.title('ARIMA Forecast of Loan Amounts')
    plt.show()


def plot_monthly_sector_trends(df):
    sector_monthly = df.groupby([pd.Grouper(freq='M'), 'sector'])['loan_amount'].sum().unstack()

    sector_monthly.plot(figsize=(15, 6))
    plt.title('Loan Amounts Over Time by Sector')
    plt.xlabel('Date')
    plt.ylabel('Loan Amount')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()


def run_time_series_analysis(df):
    df_ts = prepare_time_series(df)
    plot_loan_amount_over_time(df_ts)
    forecast_loan_amount_arima(df_ts)
    plot_monthly_sector_trends(df_ts)


# Lastly, save the cleaned data to .csv
def save_data(df, filename="cleaned_kiva_loans.csv"):
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to" '{filename}')


def main():
    df = load_data("masked_kiva_loans.csv")
    df = clean_data(df)
    df = apply_gender_split(df)
    # save_data(df)
    run_funded_amount_prediction(df)
    run_time_series_analysis(df)


#     visualize_loan_vs_funding(df)


if __name__ == "__main__":
    main()
