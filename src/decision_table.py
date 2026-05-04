import pandas as pd

class DecisionTable:
    def __init__(self, df: pd.DataFrame, universe_column: str, decision_attribute: str, conditional_columns: list[str] = None):
        self.df = df
        self.u_col = universe_column
        self.d_col = decision_attribute

        if conditional_columns is None:
            self.c_cols = [col for col in df.columns if col not in [universe_column, decision_attribute]]
        else:
            self.c_cols = conditional_columns

        if self.u_col not in df.columns:
            raise ValueError("u_col must be a subset of df columns")
        if self.d_col not in df.columns:    
            raise ValueError("d_col must be a subset of df columns")
        if not set(self.c_cols).issubset(set(df.columns)):
            raise ValueError("a_cols must be a subset of df columns")   
        
    def get_universe(self):
        return self.df[self.u_col].unique()

    def get_decision_attribute(self):
        return self.d_col

    def get_conditionals(self):
        return self.c_cols

    def get_equivalence_classes(self, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            if not set(conditionals).issubset(set(self.c_cols)):
                raise ValueError("conditionals must be a subset of the defined conditional columns")

        unique_combinations = self.df[conditionals].drop_duplicates()
        abstraction_classes = []
        for _, row in unique_combinations.iterrows():
            condition = (self.df[conditionals] == row[conditionals]).all(axis=1)
            abstraction_class = set(self.df[self.u_col][condition])
            abstraction_classes.append(abstraction_class)

        return abstraction_classes
    