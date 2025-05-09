import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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

# Lastly, save the cleaned data to .csv
def save_data(df, filename="cleaned_kiva_loans.csv"):
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved to" '{filename}')
def main():
    df = load_data("masked_kiva_loans.csv")
    df = clean_data(df)
    df = apply_gender_split(df)
    save_data(df)
#     visualize_loan_vs_funding(df)


if __name__ == "__main__":
    main()
