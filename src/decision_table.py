from itertools import combinations

import pandas as pd

from enum import Enum, auto
class Definability(Enum):
    DEFINABLE = auto() # Odpowiada Twojemu C-dokładny (Exact set)
    ROUGH = auto()

class DefinibilityType(Enum):
    ROUGHLY_DEFINABLE = auto()        
    INTERNALLY_NON_DEFINABLE = auto() 
    EXTERNALLY_NON_DEFINABLE = auto() 
    TOTALLY_NON_DEFINABLE = auto()    

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
            self._check_conditionals(conditionals)

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
            self._check_conditionals(conditionals)

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
            self._check_conditionals(conditionals)

        equivalence_classes = self.get_equivalence_classes(conditionals)
        upper_approximation = set()
        for eq_class in equivalence_classes:
            if eq_class.intersection(subset):
                upper_approximation.update(eq_class)

        return upper_approximation
    
    def get_positive_region(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        return self.get_lower_approximation(subset, conditionals)
    
    def get_boundary_region(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        lower_approximation = self.get_lower_approximation(subset, conditionals)
        upper_approximation = self.get_upper_approximation(subset, conditionals)
        return upper_approximation - lower_approximation
    
    def get_negative_region(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        upper_approximation = self.get_upper_approximation(subset, conditionals)
        universe = set(self.get_universe())
        return universe - upper_approximation
    
    def get_set_definability(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        lower_approximation = self.get_lower_approximation(subset, conditionals)
        upper_approximation = self.get_upper_approximation(subset, conditionals)

        if lower_approximation == upper_approximation:
            return Definability.DEFINABLE
        else:
            return Definability.ROUGH
        
    def get_definibility_type(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        lower_approximation = self.get_lower_approximation(subset, conditionals)
        upper_approximation = self.get_upper_approximation(subset, conditionals)

        universe = set(self.get_universe())
        if lower_approximation and upper_approximation != universe:
            return DefinibilityType.ROUGHLY_DEFINABLE
        elif not lower_approximation and upper_approximation != universe:
            return DefinibilityType.INTERNALLY_NON_DEFINABLE
        elif lower_approximation and upper_approximation == universe:
            return DefinibilityType.EXTERNALLY_NON_DEFINABLE
        else:
            return DefinibilityType.TOTALLY_NON_DEFINABLE
        
    def get_aproximation_quality(self, subset: set, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        lower_approximation = self.get_lower_approximation(subset, conditionals)
        upper_approximation = self.get_upper_approximation(subset, conditionals)

        if not upper_approximation:
            return 0.0
        return len(lower_approximation) / len(upper_approximation)

    def get_subset_family_lower_approximations(self, subsets: dict, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        # Iterujemy po items(), zachowując oryginalny klucz (np. 'X1')
        return {
            key: self.get_lower_approximation(subset, conditionals) 
            for key, subset in subsets.items()
        }.values()

    def get_subset_family_upper_approximations(self, subsets: dict, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        return {
            key: self.get_upper_approximation(subset, conditionals) 
            for key, subset in subsets.items()
        }.values()
    
    def get_subset_family_positive_region(self, subsets: dict, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        return set().union(*(self.get_positive_region(subset, conditionals) for subset in subsets.values()))
    
    def get_subset_family_quality_of_approximations(self, subsets: dict, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        positive_region = self.get_subset_family_positive_region(subsets, conditionals)
        universe = set(self.get_universe())
        if not universe:
            return 0.0
        return len(positive_region) / len(universe)
    
    def get_subset_family_accuracy_of_approximations(self, subsets: dict, conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        sum_of_lenghts_of_upper_approximations = sum(len(self.get_upper_approximation(subset, conditionals)) for subset in subsets.values())
        if sum_of_lenghts_of_upper_approximations == 0:
            return 0.0
        positive_region = self.get_subset_family_positive_region(subsets, conditionals)
        return len(positive_region) / sum_of_lenghts_of_upper_approximations
    
    def get_subsets_family_dependability(self, subsets: list[set], conditionals: list[str] = None):
        if conditionals is None:
            conditionals = self.c_cols
        else:
            self._check_conditionals(conditionals)

        positive_region = self.get_subset_family_positive_region(subsets, conditionals)
        universe = set(self.get_universe())
        if not universe:
            return 0.0
        return len(positive_region) / len(universe)

    def is_subset_independent(self, conditional_subset: list[str] = None):
        conditional_subset = self._handle_conditionals(conditional_subset)

        equivalence_classes = self.get_equivalence_classes(conditional_subset)  

        for attr in conditional_subset:
            reduced_subset = list(set(conditional_subset) - {attr})
            subset_equivalence_classes = self.get_equivalence_classes(reduced_subset)
            if equivalence_classes == subset_equivalence_classes:
                return False  
            
        return True

    def is_subset_conditionally_independent(self, conditional_relation: dict, conditionals_subset: list[str] = None):
        conditionals_subset = self._handle_conditionals(conditionals_subset)

        positive_region = self.get_subset_family_positive_region(conditional_relation, conditionals_subset)

        for attr in conditionals_subset:
            reduced_subset = list(set(conditionals_subset) - {attr})
            reduced_positive_region = self.get_subset_family_positive_region(conditional_relation, reduced_subset)
            if positive_region == reduced_positive_region:
                return False    
        
        return True
    
    def get_attribute_reducts(self, conditionals_subset: list[str] = None):
        conditionals_subset = self._handle_conditionals(conditionals_subset)

        if self.is_subset_independent(conditionals_subset):
            return None

        equivalence_class = self.get_equivalence_classes(conditionals_subset)
        reduct_list = []


        for attr in conditionals_subset:
            reduced_subset = list(set(conditionals_subset) - {attr})
            
            reduced_equivalence_class = self.get_equivalence_classes(reduced_subset)
            if equivalence_class == reduced_equivalence_class and self.is_subset_independent(reduced_subset):
                reduct_list.append(reduced_subset)
            
        return reduct_list
    
    def get_relative_reducts(self, conditional_relation: dict, conditionals_subset: list[str] = None):
        conditionals_subset = self._handle_conditionals(conditionals_subset)

        if self.is_subset_conditionally_independent(conditional_relation, conditionals_subset):
            return None

        equivalence_class = self.get_equivalence_classes(conditionals_subset)
        reduct_list = []

        for attr in conditionals_subset:
            reduced_subset = list(set(conditionals_subset) - {attr})
            
            reduced_eqivalence_class = self.get_equivalence_classes(reduced_subset)
            if equivalence_class == reduced_eqivalence_class and self.is_subset_conditionally_independent(conditional_relation, reduced_subset):
                reduct_list.append(reduced_subset)
            
        return reduct_list

    def _handle_conditionals(self, conditionals):
        if conditionals is None:
            return self.c_cols
        else:
            self._check_conditionals(conditionals)
            return conditionals

    def _check_conditionals(self, conditionals):
        if not set(conditionals).issubset(set(self.c_cols)):
            raise ValueError("Conditionals must be a subset of the defined conditional columns")
        
    
        