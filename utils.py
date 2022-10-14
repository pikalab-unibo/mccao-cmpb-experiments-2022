import re
from typing import Callable
import numpy as np
from psyki.logic.datalog import DatalogFormula, Expression
from psyki.logic.datalog.grammar import optimize_datalog_formula
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout


def string_var_compliant(string: str) -> str:
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string[0].upper() + string[1:]
    return re.sub(r'[ ]([a-z])', lambda match: match.group(1).capitalize(), string).replace(' ', '')


def create_nn(input_size: int = 815, neurons_per_layer=None) -> Model:
    if neurons_per_layer is None:
        neurons_per_layer = list([8, 4, 1])
    input_layer = Input((input_size,))
    x = input_layer
    for neurons in neurons_per_layer[:-1]:
        x = Dense(neurons, activation='relu')(x)
        x = Dropout(0.2)(x)
    x = Dense(neurons_per_layer[-1], activation='sigmoid')(x)
    network = Model(input_layer, x)
    network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
    return network


def formula_to_callable(formula: DatalogFormula) -> Callable:
    optimize_datalog_formula(formula)
    rhs = formula.rhs
    if isinstance(rhs, Expression):
        if len(rhs.nary) > 0:
            result: Callable = lambda x: np.all([x[clause.lhs.name.lower()] > clause.rhs.value if clause.op == '>' else x[clause.lhs.name.lower()] <= clause.rhs.value for clause in rhs.nary], axis=0)
        elif isinstance(rhs.lhs, Expression) and isinstance(rhs.rhs, Expression):
            e1, e2 = rhs.lhs, rhs.rhs
            result_e1: Callable = formula_to_callable(e1)
            result_e2: Callable = formula_to_callable(e2)
            result: Callable = lambda x: np.all([result_e1(x), result_e2(x)], axis=0)
        else:
            result: Callable = lambda x: x[rhs.lhs.name.lower()] > rhs.rhs.value if rhs.op == '>' else x[rhs.lhs.name.lower()] <= rhs.rhs.value
    else:
        if isinstance(formula.lhs, Expression) and isinstance(formula.rhs, Expression):
            e1, e2 = formula.lhs, formula.rhs
            result_e1: Callable = formula_to_callable(e1)
            result_e2: Callable = formula_to_callable(e2)
            result: Callable = lambda x: np.all([result_e1(x), result_e2(x)], axis=0)
        else:
            result: Callable = lambda x: x[formula.lhs.name.lower()] > formula.rhs.value if formula.op == '>' else x[formula.lhs.name.lower()] <= formula.rhs.value
    return result


def formulae_to_callable(formulae: list[DatalogFormula]) -> Callable:
    callables = [formula_to_callable(f) for f in formulae]
    return lambda x: np.any([f(x) for f in callables], axis=0)
