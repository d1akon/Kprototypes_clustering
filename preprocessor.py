import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Clase para el preprocesamiento de los datos.
    """
    def __init__(self, categorical_cols, numeric_cols):
        """
        :param categorical_cols: Lista de columnas categóricas.
        :param numeric_cols: Lista de columnas numéricas.
        """
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame):
        """
        Ajusta y transforma los datos categóricos y numéricos.
        :param df: DataFrame de entrada.
        :return: DataFrames escalados y codificados.
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
        Transforma datos nuevos usando el escalador ajustado.
        """
        for col in self.categorical_cols:
            if df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')

        df_num = df[self.numeric_cols]
        df_cat = df[self.categorical_cols]

        df_num_scaled = pd.DataFrame(self.scaler.transform(df_num), columns=self.numeric_cols)
        df_cat_encoded = df_cat.apply(lambda x: x.cat.codes)

        return df_num_scaled, df_cat_encoded
