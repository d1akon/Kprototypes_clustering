import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Class for preprocessing the data.
    """
    def __init__(self, categorical_cols, numeric_cols):
        """
        :param categorical_cols: List of categorical columns.
        :param numeric_cols: List of numerical columns.
        """
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame):
        """
        Fits and transforms the categorical and numerical data.
        :param df: Input DataFrame.
        :return: Scaled and encoded DataFrames.
        """
        for col in self.categorical_cols:
            df[col] = df[col].astype('category')

        df_num = df[self.numeric_cols]
        df_cat = df[self.categorical_cols]

        df_num_scaled = pd.DataFrame(self.scaler.fit_transform(df_num), columns=self.numeric_cols)
        df_cat_encoded = df_cat.apply(lambda x: x.cat.codes)

        return df_num_scaled, df_cat_encoded

    def transform(self, df: pd.DataFrame):
        """
        Transforms new data using the fitted scaler.
        """
        for col in self.categorical_cols:
            if df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')

        df_num = df[self.numeric_cols]
        df_cat = df[self.categorical_cols]

        df_num_scaled = pd.DataFrame(self.scaler.transform(df_num), columns=self.numeric_cols)
        df_cat_encoded = df_cat.apply(lambda x: x.cat.codes)

        return df_num_scaled, df_cat_encoded
