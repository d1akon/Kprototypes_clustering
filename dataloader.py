import pandas as pd

class DataLoader:
    """
    Clase responsable de cargar el dataset desde una URL.
    """
    def __init__(self, url: str):
        """
        :param url: URL al archivo CSV.
        """
        self.url = url

    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset desde una URL.
        :return: DataFrame de pandas con los datos.
        """
        return pd.read_csv(self.url)
