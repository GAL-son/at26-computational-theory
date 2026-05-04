import pandas as pd

class InformationSystem:
    def __init__(self, df: pd.DataFrame, universe_column: str, attribute_columns: list[str] = None):
        """
        df: DataFrame containing data
        u_cols: column defining the universe of objects
        q_cols: list of columns defining the attributes of the information system. If empty, all columns except u_cols are considered as attributes.
        """

        self.df = df
        self.u_col = universe_column

        if attribute_columns is None:
            self.a_cols = [col for col in df.columns if col != universe_column]
        else:
            self.a_cols = attribute_columns

        if self.u_col not in df.columns:
            raise ValueError("u_col must be a subset of df columns")
        
        if not set(self.a_cols).issubset(set(df.columns)):
            raise ValueError("q_cols must be a subset of df columns")
        
    def get_universe(self):
        return self.df[self.u_col].unique()

    def get_attributes(self):
        return self.a_cols

    def get_attribute_values(self, attribute: str):
        if attribute not in self.a_cols:
            raise ValueError("Attribute must be one of the defined attribute columns")
        return self.df[attribute].unique()
    
    


