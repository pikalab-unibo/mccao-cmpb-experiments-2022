import re
from typing import Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from psyki.logic.datalog import DatalogFormula, Expression
from psyki.logic.datalog.grammar import optimize_datalog_formula, Nary
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clip, sum, epsilon


from resources.dataset import PATH as DATASET_PATH

DEFAULT_INPUT_LAYER = 16
DEFAULT_HIDDEN_LAYER = 8
DEFAULT_OUTPUT_LAYER = 1


def string_var_compliant(string: str) -> str:
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string[0].upper() + string[1:]
    return re.sub(r'[ ]([a-z])', lambda match: match.group(1).capitalize(), string).replace(' ', '')


def create_nn(input_size, neurons_per_layer=None) -> Model:
    if neurons_per_layer is None:
        neurons_per_layer = list([DEFAULT_INPUT_LAYER, DEFAULT_HIDDEN_LAYER, DEFAULT_OUTPUT_LAYER])
    input_layer = Input((input_size,))
    x = input_layer
    for neurons in neurons_per_layer[:-1]:
        x = Dense(neurons, activation='relu')(x)
    x = Dense(neurons_per_layer[-1], activation='sigmoid')(x)
    network = Model(input_layer, x)
    network.compile(optimizer='adam', metrics=['accuracy', precision], loss='binary_crossentropy')
    return network


def formula_to_callable(formula: DatalogFormula, rules: dict[str: list[DatalogFormula]]) -> Callable:

    def eval_element(clause):
        if isinstance(clause, Expression):
            return lambda x: x[clause.lhs.name] > clause.rhs.value if clause.op == '>' else x[clause.lhs.name] <= clause.rhs.value
        elif isinstance(clause, Nary):
            return lambda x: np.any([formula_to_callable(rule, rules)(x) for rule in rules[clause.name]], axis=0)
        elif isinstance(clause.rhs, Nary):
            return lambda x: np.any([formula_to_callable(rule, rules)(x) for rule in rules[clause.rhs.name]], axis=0)
        else:
            raise Exception('Not able to convert formula into filter')

    optimize_datalog_formula(formula)
    rhs = formula.rhs
    if isinstance(rhs, Expression):
        if len(rhs.nary) > 0:
            result: Callable = lambda x: np.all([eval_element(clause)(x) for clause in rhs.nary], axis=0)
        elif isinstance(rhs.lhs, Expression) and isinstance(rhs.rhs, Expression):
            e1, e2 = rhs.lhs, rhs.rhs
            result_e1: Callable = formula_to_callable(e1)
            result_e2: Callable = formula_to_callable(e2)
            result: Callable = lambda x: np.all([result_e1(x), result_e2(x)], axis=0)
        else:
            result: Callable = eval_element(rhs)
    else:
        if isinstance(formula.lhs, Expression) and isinstance(formula.rhs, Expression):
            e1, e2 = formula.lhs, formula.rhs
            result_e1: Callable = formula_to_callable(e1)
            result_e2: Callable = formula_to_callable(e2)
            result: Callable = lambda x: np.all([result_e1(x), result_e2(x)], axis=0)
        else:
            result: Callable = eval_element(formula)
    return result


def formulae_to_callables(formulae: list[DatalogFormula]) -> list[Callable]:
    predicates = list(set([formula.lhs.predication for formula in formulae if formula.lhs.predication != 'target']))
    kb = {}
    for predicate in predicates:
        for formula in formulae:
            if formula.lhs.predication == predicate:
                if predicate not in kb.keys():
                    kb[predicate] = [formula]
                else:
                    kb[predicate] = kb[predicate] + [formula]
    classification_formulae = list([formula for formula in formulae if formula.lhs.predication == 'target'])
    callables = [formula_to_callable(f, kb) for f in classification_formulae]
    return callables


def data_to_struct(data: pd.Series):
    from tuprolog.core import numeric, var, struct

    head = 'target'
    terms = [numeric(item) for item in data]
    terms.append(var('X'))
    return struct(head, terms)


def get_ingredients(files: list[str]):
    from pandas import read_csv
    ingredients = []
    for file in files:
        new_ingredients = [string_var_compliant(i) for i in list(read_csv(DATASET_PATH / file).iloc[:, 0])]
        new_ingredients = [i if i not in ingredients else 'Compound' + i for i in new_ingredients]
        ingredients += list(set(new_ingredients))
    return sorted(ingredients)


def get_ingredients_id_map(files: list[str]):
    from pandas import read_csv
    ingredients, indices = [], []
    for file in files:
        df = read_csv(DATASET_PATH / file)
        new_ingredients = [string_var_compliant(i) for i in list(df.iloc[:, 0])]
        new_ingredients = [i if i not in ingredients else 'Compound' + i for i in new_ingredients]
        ingredients += new_ingredients
        indices += list(df['Entity ID']) if 'Entity ID' in df.columns else list(df['entity_id'])
    ingredients = [string_var_compliant(ingredient) for ingredient in ingredients]
    return {k: v for k, v in zip(indices, ingredients)}


def get_categories_ingredients_map(files: list[str]):
    from pandas import read_csv
    categories_ingredients: dict[str:list[str]] = {}
    for file in files:
        df = read_csv(DATASET_PATH / file)
        new_categories = list(set(df['Category']))
        for category in new_categories:
            new_ingredients = list(df.loc[df['Category'] == category].iloc[:, 0])
            new_ingredients = [string_var_compliant(i) for i in new_ingredients]
            if category in categories_ingredients.keys():
                ingredients = categories_ingredients[category]
                new_ingredients = [i if i not in ingredients else 'Compound' + i for i in new_ingredients]
                categories_ingredients[category] = ingredients + new_ingredients
            else:
                categories_ingredients[category] = new_ingredients
    return categories_ingredients


def get_liked_recipes(theory, dataset, like=True):
    label = 'positive' if like else 'negative'
    user_preferences_formulae = prolog_to_datalog(theory)
    user_preferences_formulae = [f for f in user_preferences_formulae if f.lhs.arg.last.name == label]
    preferences_filters = formulae_to_callables(user_preferences_formulae)
    preferences_filter: Callable = lambda x: np.any([f(x) for f in preferences_filters], axis=0)
    liked_recipes = dataset[dataset.apply(preferences_filter, axis=1)]
    return liked_recipes


def tp(y_true, y_pred):
    return sum(tf.round(clip(y_true * y_pred, 0, 1)))


def recall(y_true, y_pred):
    true_positives = tp(y_true, y_pred)
    possible_positives = sum(tf.round(clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = tp(y_true, y_pred)
    predicted_positives = sum(tf.round(clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())
    return precision


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+epsilon()))


def formula_to_ingredients(formula):
    body = formula.rhs
    has, not_has = [], []
    if len(body.nary) > 0:
        for element in body.nary:
            if element.op == '=<':
                not_has.append(element.lhs.name)
            else:
                has.append(element.lhs.name)
    return has, not_has


def formula_size(formula):
    body = formula.rhs
    if len(body.nary) > 0:
        return len(body.nary)
    elif body.rhs is not None:
        return 2
    else:
        return 1


def sort_formulae_by_size(formulae):
    for formula in formulae:
        optimize_datalog_formula(formula)
    return sorted(formulae, key=lambda x: formula_size(x), reverse=False)
