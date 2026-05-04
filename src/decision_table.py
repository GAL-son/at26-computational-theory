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

        self.cache_enabled = False
        self.equivalence_classes_cache = None

    def enable_cache(self):
        self.cache_enabled = True
        
    def get_universe(self):
        return self.df[self.u_col].unique()

    def get_decision_attribute(self):
        return self.d_col
    
    def get_decision_values(self):
        return self.df[self.d_col].unique()

    def get_conditionals(self):
        return self.c_cols

    def get_equivalence_classes(self, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            if not set(conditionals).issubset(set(self.c_cols)):
                raise ValueError("conditionals must be a subset of the defined conditional columns")

        if self.cache_enabled and self.equivalence_classes_cache is not None:
            cached_result = self.equivalence_classes_cache.get(hash(frozenset(conditionals)))
            if cached_result is not None:
                return cached_result

        unique_combinations = self.df[conditionals].drop_duplicates()
        abstraction_classes = []
        for _, row in unique_combinations.iterrows():
            condition = (self.df[conditionals] == row[conditionals]).all(axis=1)
            abstraction_class = set(self.df[self.u_col][condition])
            abstraction_classes.append(abstraction_class)

        if self.cache_enabled:
            if self.equivalence_classes_cache is None:
                self.equivalence_classes_cache = {}
            self.equivalence_classes_cache[hash(frozenset(conditionals))] = abstraction_classes

        return abstraction_classes
    
    def get_objects_with_decision_value(self, decision_value):
        return set(self.df[self.u_col][self.df[self.d_col] == decision_value])

    def get_lower_approximation(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            if not set(conditionals).issubset(set(self.c_cols)):
                raise ValueError("conditionals must be a subset of the defined conditional columns")

        equivalence_classes = self.get_equivalence_classes(conditionals)
        lower_approximation = set()
        for eq_class in equivalence_classes:
            if eq_class.issubset(subset):
                lower_approximation.update(eq_class)

        return lower_approximation

    def get_upper_approximation(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            if not set(conditionals).issubset(set(self.c_cols)):
                raise ValueError("conditionals must be a subset of the defined conditional columns")

        equivalence_classes = self.get_equivalence_classes(conditionals)
        upper_approximation = set()
        for eq_class in equivalence_classes:
            if eq_class.intersection(subset):
                upper_approximation.update(eq_class)

        return upper_approximation
    