import distutils.cmd
import random
from distutils.core import setup
import numpy as np
import pandas as pd
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog
from setuptools import find_packages
from resources.dataset import PATH as DATASET_PATH
from resources.models import PATH as MODEL_PATH
from resources.rules import PATH as RULES_PATH
from utils import create_nn


EPOCHS: int = 10


class GenerateUsersScores(distutils.cmd.Command):
    description = 'generate a csv file with the users\' scores'
    user_options = [('file=', 'f', 'output file name (default is users_scores)'),
                    # ('seed=', 's', 'seed for reproducibility, (default is 0)'),
                    ('ingredient=', 'i', 'file name to get all ingredients (default is recipes)'),
                    ('hints=', 'h', 'file name of users\' ingredient preferences (default users_preferences'),
                    ('min=', 'm', 'minimum value of a preference, (default is -1'),
                    ('max=', 'M', 'maximum value of a preference, (default is 1')]
    dataset_path = DATASET_PATH
    default_output_file_name: str = 'users_scores'
    default_hints_file: str = 'users_preferences'
    default_ingredient: str = 'recipes'
    default_min: float = -1.
    default_max: float = 1.

    def initialize_options(self) -> None:
        self.file = self.default_output_file_name
        self.ingredient = self.default_ingredient
        self.hints = self.default_hints_file
        self.min = self.default_min
        self.max = self.default_max

    def finalize_options(self) -> None:
        self.min = float(self.min)
        self.max = float(self.max)
        self.file = self.file
        self.hints = self.hints
        self.ingredient = self.ingredient

    def run(self) -> None:
        from numpy import arange
        from pandas import read_csv, DataFrame

        ingredient_list = read_csv(self.dataset_path / (self.ingredient + '.csv')).columns
        users_preferences = read_csv(self.dataset_path / (self.hints + '.csv'))
        ingredient_sublist = users_preferences.columns
        ingredient_scores = DataFrame(0., index=arange(len(users_preferences.index)), columns=ingredient_list)
        for j, user_preferences in users_preferences.iterrows():
            for i, score in enumerate(user_preferences):
                if ingredient_sublist[i] in ingredient_list:
                    ingredient_scores.at[j, ingredient_sublist[i]] = score
        ingredient_scores.to_csv(self.dataset_path / (self.file + '.csv'), index=False)


class GenerateDataset(distutils.cmd.Command):
    description = 'generate the labeled dataset'
    user_options = []
    dataset_path = DATASET_PATH
    recipes = 'recipes'
    users = 'users_scores'
    seed = 0

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import read_csv, DataFrame

        random.seed(self.seed)
        recipes = read_csv(self.dataset_path / (self.recipes + '.csv'))
        users = read_csv(self.dataset_path / (self.users + '.csv'))
        rxu = recipes @ users.T
        random_choise = DataFrame([random.random() > 0.95 for _ in range(rxu.shape[0])])
        labels = ((rxu > rxu.mean()) | ((rxu >= 0) & random_choise)).astype(int)
        dataset = recipes.join(labels)
        dataset.columns = list(recipes.columns) + list(['target', ])
        # dataset = dataset.iloc[:, list(range(1, len(recipes.columns), 2)) + list([len(recipes.columns)])]
        dataset.to_csv(self.dataset_path / 'nn_dataset_furkan_user.csv', index=False)


class GenerateUsersPreferences(distutils.cmd.Command):
    description = 'generate a csv file with the users\' preferences'
    user_options = []
    dataset_path = DATASET_PATH
    seed = 0
    default_file: str = 'users_preferences'
    min_n, max_n = -1., 1.
    min_v, max_v = 1., 10.

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import DataFrame
        from resources.nutrition_styles import NUTRITION_STYLES, NUTRITION_USERS

        # Furkan's data
        styles = NUTRITION_STYLES
        users = NUTRITION_USERS

        columns = users['furkan']['preferences'].keys()
        scores = []
        np.random.seed(self.seed)
        for _, (min_limit, max_limit) in users['furkan']['preferences'].items():
            scores.append(round(np.random.uniform(low=min_limit - 0.5, high=max_limit + 0.5)))
        df = DataFrame([scores], columns=columns)
        df = (df - self.min_v) * (self.max_n - self.min_n) / (self.max_v - self.min_v) + self.min_n
        df.to_csv(self.dataset_path / (self.default_file + '.csv'), index=False)


class TrainNN(distutils.cmd.Command):
    description = 'create and train a NN on Furkan dataset'
    user_options = []
    dataset_path = DATASET_PATH
    model_path = MODEL_PATH
    dataset_name = 'nn_dataset_furkan_user'
    neurons_per_layer = [64, 32, 1]
    seed = 0
    n_epochs = EPOCHS
    batch_size = 32

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import read_csv
        from sklearn.model_selection import train_test_split
        from tensorflow.python.framework.random_seed import set_seed

        set_seed(self.seed)
        dataset = read_csv(self.dataset_path / (self.dataset_name + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=2 / 3, random_state=self.seed, stratify=dataset.iloc[:, -1])
        input_size = dataset.shape[1]-1
        network = create_nn(input_size)
        network.summary()
        network.fit(train.iloc[:, :-1], train.iloc[:, -1:], epochs=self.n_epochs, batch_size=self.batch_size)
        loss, accuracy = network.evaluate(test.iloc[:, :-1], test.iloc[:, -1:])
        print('Accuracy: ' + str(accuracy))
        network.save(self.model_path / ('furkan_network' + '.h5'))


class ExtractRules(distutils.cmd.Command):
    description = 'create and train a NN on Furkan dataset'
    user_options = []
    seed = 0
    dataset_path = DATASET_PATH
    model_path = MODEL_PATH
    rules_path = RULES_PATH
    simplify = False
    dataset_name = 'nn_dataset_furkan_user'
    mapping = {'negative': 0, 'positive': 1}
    inverse_mapping = {0: 'negative', 1: 'positive'}

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from psyke import Extractor
        from psyke.utils.logic import pretty_clause
        from pandas import read_csv, DataFrame
        from sklearn.model_selection import train_test_split
        from tensorflow.python.framework.random_seed import set_seed
        from tensorflow.python.keras.models import load_model
        from tuprolog.solve.prolog import prolog_solver

        set_seed(self.seed)
        dataset = read_csv(self.dataset_path / (self.dataset_name + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=2/3, random_state=self.seed, stratify=dataset.iloc[:, -1])
        train['target'] = train['target'].apply(lambda x: 'positive' if x == 1 else 'negative')
        network = load_model(self.model_path / ('furkan_network' + '.h5'))
        network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        extractor = Extractor.cart(network, simplify=self.simplify, max_depth=10, max_leaves=30)
        theory = extractor.extract(train, self.mapping)
        # rules = prune(simplify(rules))
        solver = prolog_solver(static_kb=theory)
        subset = test  # DataFrame(test.iloc[0:100, :])
        predicted = network.predict(subset.iloc[:, :-1])
        predicted = DataFrame(predicted)[0].apply(lambda x: 'negative' if x < 0.5 else 'positive')
        with open(self.rules_path / 'furkan_rules.csv', 'w') as file:
            for rule in theory.clauses:
                file.write(pretty_clause(rule)+'.\n')


class InjectRules(distutils.cmd.Command):
    description = 'create and train a NN on Furkan dataset using the extracted rules'
    user_options = []
    rules_path = RULES_PATH
    model_path = MODEL_PATH
    dataset_path = DATASET_PATH
    dataset_name = 'nn_dataset_furkan_user'
    seed = 0
    n_epochs = EPOCHS
    batch_size = 32

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import read_csv
        from sklearn.model_selection import train_test_split
        from tensorflow.python.framework.random_seed import set_seed
        from psyki.ski import Injector

        set_seed(self.seed)
        dataset = read_csv(self.dataset_path / (self.dataset_name + '.csv')).astype(int)
        feature_mapping = {feature.capitalize(): i for i, feature in enumerate(dataset.columns)}
        train, test = train_test_split(dataset, train_size=2 / 3, random_state=self.seed, stratify=dataset.iloc[:, -1])
        input_size = dataset.shape[1] - 1
        network = create_nn(input_size)
        injector = Injector.kins(network, feature_mapping)
        print('Retrieve Prolog theory')
        prolog_theory = file_to_prolog(self.rules_path / 'furkan_rules.csv')
        print('Convert to Datalog')
        knowledge = prolog_to_datalog(prolog_theory)
        print('Injection')
        new_predictor = injector.inject(knowledge)
        new_predictor.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        print('Training')
        new_predictor.fit(train.iloc[:, :-1], train.iloc[:, -1:], epochs=self.n_epochs, batch_size=self.batch_size)
        print('Evaluate')
        loss, accuracy = new_predictor.evaluate(test.iloc[:, :-1], test.iloc[:, -1:])
        print('Accuracy: ' + str(accuracy))
        network.save(self.model_path / ('furkan_network_with_injection' + '.h5'))


def data_to_struct(data: pd.Series):
    from tuprolog.core import numeric, var, struct

    head = data.keys()[-1]
    terms = [numeric(item) for item in data[:-1]]
    terms.append(var('X'))
    return struct(head, terms)


setup(
    name='scripts',  # Required
    description='Script to work with datasets and SKE/SKI for Expectation',
    license='Apache 2.0 License',
    long_description_content_type='text/markdown',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    zip_safe=False,
    platforms="Independant",
    cmdclass={
        'generate_users_scores': GenerateUsersScores,
        'generate_users_preferences': GenerateUsersPreferences,
        'generate_dataset': GenerateDataset,
        'build_and_train_nn': TrainNN,
        'extract_rules': ExtractRules,
        'inject_rules': InjectRules,
    },
)
