import numpy as np
import pandas as pandas


def data_cleaning(df, if_test=False):
    df.loc[df['age'] < 10, 'age'] = df['age'].median()
    df.loc[df['MonthlyIncome'].isna(), 'MonthlyIncome'] = df['MonthlyIncome'].mean()
    df.loc[df['NumberOfDependents'].isna(), 'NumberOfDependents'] = df['NumberOfDependents'].mode()[0]

    if if_test:
        df = df.drop(['SeriousDlqin2yrs'], axis=1)

    return df
