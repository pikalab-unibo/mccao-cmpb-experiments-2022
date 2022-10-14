import distutils.cmd
import random
import sys
from distutils.core import setup
from os import system
import numpy as np
import pandas as pd
from pandas import Series
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog
from setuptools import find_packages
from resources.dataset import PATH as DATASET_PATH
from resources.models import PATH as MODEL_PATH
from resources.rules import PATH as RULES_PATH
from resources.prescriptions import PATH as PRESCRIPTION_PATH
from utils import create_nn, formulae_to_callable, string_var_compliant

EPOCHS: int = 10


class DownloadDatasets(distutils.cmd.Command):

    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        system('curl -o culinary_db.zip --url https://cosylab.iiitd.edu.in/culinarydb/static/data/CulinaryDB.zip')
        system('unzip culinary_db.zip -d ' + str(DATASET_PATH))
        system('rm -rfd culinary_db.zip')


class GenerateUsersScores(distutils.cmd.Command):
    description = 'generate a csv file with the users\' scores'
    user_options = [('file=', 'f', 'output file name (default is users_scores)'),
                    # ('seed=', 's', 'seed for reproducibility, (default is 0)'),
                    ('hints=', 'h', 'file name of users\' ingredient preferences (default users_preferences'),
                    ('min=', 'm', 'minimum value of a preference, (default is -1'),
                    ('max=', 'M', 'maximum value of a preference, (default is 1')]
    dataset_path = DATASET_PATH
    default_output_file_name: str = 'user_scores'
    default_hints_file: str = 'user_preferences'
    default_min: float = -1.
    default_max: float = 1.

    def initialize_options(self) -> None:
        self.file = self.default_output_file_name
        self.ingredients = '02_Ingredients'
        self.compound_ingredients = '03_Compound_Ingredients'
        self.hints = self.default_hints_file
        self.min = self.default_min
        self.max = self.default_max

    def finalize_options(self) -> None:
        self.min = float(self.min)
        self.max = float(self.max)
        self.file = self.file
        self.hints = self.hints
        self.ingredients = self.ingredients

    def run(self) -> None:
        from numpy import arange
        from pandas import read_csv, DataFrame

        ingredients = get_ingredients([self.ingredients, self.compound_ingredients])
        users_preferences = read_csv(self.dataset_path / (self.hints + '.csv'))
        ingredients_sublist = users_preferences.columns
        ingredients_sublist = sorted([string_var_compliant(ingredient) for ingredient in ingredients_sublist])
        ingredients_scores = DataFrame(0., index=arange(len(users_preferences.index)), columns=ingredients)
        for j, user_preferences in users_preferences.iterrows():
            for i, score in enumerate(user_preferences):
                if string_var_compliant(ingredients_sublist[i]) in ingredients:
                    ingredients_scores.at[j, ingredients_sublist[i]] = score
        ingredients_scores.to_csv(self.dataset_path / (self.file + '.csv'), index=False)


class GenerateDataset(distutils.cmd.Command):
    description = 'generate the labeled dataset'
    user_options = []
    dataset_path = DATASET_PATH
    ingredients = '02_Ingredients'
    compound_ingredients = '03_Compound_Ingredients'
    recipes = '04_Recipe-Ingredients_Aliases'
    users = 'user_scores'
    seed = 0

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import read_csv, DataFrame

        random.seed(self.seed)
        ingredients_list = get_ingredients([self.ingredients, self.compound_ingredients])
        ingredients = get_ingredients_id_map([self.ingredients, self.compound_ingredients])
        recipes = read_csv(self.dataset_path / (self.recipes + '.csv'))
        matrix_recipes_ingredients = np.zeros(shape=(len(set(recipes.iloc[:, 0])), len(ingredients_list)))
        for i, recipe in enumerate(set(recipes.iloc[:, 0])):
            local_ingredients = list(recipes.loc[recipes['Recipe ID'] == recipe].iloc[:, -1])
            local_ingredients = [string_var_compliant(ingredients[i]) for i in local_ingredients if i in ingredients.keys()]
            for ingredient in local_ingredients:
                matrix_recipes_ingredients[i, ingredients_list.index(ingredient)] = 1
        recipes = DataFrame(matrix_recipes_ingredients, columns=ingredients_list)
        users = read_csv(self.dataset_path / (self.users + '.csv'))
        rxu = np.dot(recipes, users.T)
        random_choise = DataFrame([random.random() > 0.95 for _ in range(rxu.shape[0])])
        labels = ((rxu > rxu.mean()) | ((rxu >= 1) & random_choise)).astype(int)
        dataset = recipes.join(labels)
        dataset.columns = list(recipes.columns) + list(['target', ])
        dataset.to_csv(self.dataset_path / 'nn_dataset_furkan_user.csv', index=False)


class GenerateUsersPreferences(distutils.cmd.Command):
    description = 'generate a csv file with the users\' preferences'
    user_options = []
    dataset_path = DATASET_PATH
    ingredients = '02_Ingredients'
    compound_ingredients = '03_Compound_Ingredients'
    seed = 0
    default_file: str = 'user_preferences'
    min_n, max_n = -1., 1.
    min_v, max_v = 1., 10.

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import DataFrame, read_csv
        from resources.nutrition_styles import NUTRITION_STYLES, NUTRITION_USERS

        def compare_series_with_var_string(items: Series, string: str) -> Series:
            return Series([string_var_compliant(item) == string for item in items])

        # Furkan's data
        styles = NUTRITION_STYLES
        users = NUTRITION_USERS

        ingredients = read_csv(self.dataset_path / (self.ingredients + '.csv'))
        compound_ingredients = read_csv(self.dataset_path / (self.compound_ingredients + '.csv'))
        scores = {}
        np.random.seed(self.seed)
        for key, (min_limit, max_limit) in users['furkan']['category_preferences'].items():
            bool_filter = compare_series_with_var_string(ingredients['Category'], key)
            local_ingredients = list(ingredients.loc[bool_filter].iloc[:, 0])
            bool_filter = compare_series_with_var_string(compound_ingredients['Category'], key)
            if any(bool_filter):
                local_ingredients += list(compound_ingredients.loc[bool_filter].iloc[:, 0])
            for ingredient in local_ingredients:
                scores[string_var_compliant(ingredient)] = round(np.random.uniform(low=min_limit - 0.5, high=max_limit + 0.5))
        for key, (min_limit, max_limit) in users['furkan']['ingredient_preferences'].items():
            scores[string_var_compliant(key)] = round(np.random.uniform(low=min_limit - 0.5, high=max_limit + 0.5))
        scores = dict(sorted(scores.items()))
        df = DataFrame.from_records([scores])
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
        input_size = dataset.shape[1] - 1
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
        from pandas import read_csv
        from sklearn.model_selection import train_test_split
        from tensorflow.python.framework.random_seed import set_seed
        from tensorflow.python.keras.models import load_model

        set_seed(self.seed)
        dataset = read_csv(self.dataset_path / (self.dataset_name + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=2 / 3, random_state=self.seed, stratify=dataset.iloc[:, -1])
        train['target'] = train['target'].apply(lambda x: 'positive' if x == 1 else 'negative')
        network = load_model(self.model_path / ('furkan_network' + '.h5'))
        network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        extractor = Extractor.cart(network, simplify=self.simplify, max_depth=10, max_leaves=30)
        theory = extractor.extract(train, self.mapping)
        with open(self.rules_path / 'furkan_rules.csv', 'w') as file:
            for rule in theory.clauses:
                file.write(pretty_clause(rule) + '.\n')


class GenerateCommonKB(distutils.cmd.Command):

    user_options = []
    ingredients = '02_Ingredients'
    compound_ingredients = '03_Compound_Ingredients'

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        kb = ''
        ingredients = get_ingredients([self.ingredients, self.compound_ingredients])
        categories_ingredients = get_categories_ingredients_map([self.ingredients, self.compound_ingredients])
        for k, vs in categories_ingredients.items():
            k = string_var_compliant(k)
            k = k[0].lower() + k[1:]
            for v in vs:
                kb += k + '(' + ', '.join(ingredients) + ') :-\n\t' + v + ' > 0.5.\n'
            kb += '\n'
        with open(RULES_PATH / 'kb.csv', "w") as file:
            file.write(kb)


class ProposeRecipes(distutils.cmd.Command):
    description = 'For the moment just print the number of recipes satisfying both user preferences and prescriptions'
    user_options = []
    dataset_path = DATASET_PATH
    rules_path = RULES_PATH
    prescriptions_path = PRESCRIPTION_PATH
    prescriptions_name = 'nutrition-plan-lunch.csv'
    user_preferences = 'furkan_rules.csv'
    dataset_name = 'recipes.csv'
    recipes_data = 'recipes_full_data.csv'
    kb = 'kb.csv'

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from pandas import read_csv

        data = read_csv(self.dataset_path / self.dataset_name).astype(int)
        user_preferences_theory = file_to_prolog(self.rules_path / self.user_preferences)
        sys.setrecursionlimit(2000)
        user_preferences_formulae = prolog_to_datalog(user_preferences_theory)
        user_preferences_formulae = [f for f in user_preferences_formulae if f.lhs.arg.last.name == 'positive']
        preferences_filters = formulae_to_callable(user_preferences_formulae)

        kb = file_to_prolog(self.rules_path / self.kb)
        kb_formulae = prolog_to_datalog(kb)
        prescriptions_theory = file_to_prolog(self.prescriptions_path / self.prescriptions_name)
        prescriptions_formulae = prolog_to_datalog(prescriptions_theory)
        prescriptions_formulae = kb_formulae + prescriptions_formulae
        prescriptions_filters = formulae_to_callable(prescriptions_formulae)

        preferred_recipes = data[data.apply(preferences_filters, axis=1)]
        prescriptions_recipes = data[data.apply(prescriptions_filters, axis=1)]
        preferred_and_prescribed_recipes = preferred_recipes[preferred_recipes.apply(prescriptions_filters, axis=1)]

        print("Recipes accepted according to user's preferences: " + str(preferred_recipes.shape[0]))
        print("Recipes compliant with prescriptions: " + str(prescriptions_recipes.shape[0]))
        print("Recipes compliant to both prescriptions and user's preferences: " + str(preferred_and_prescribed_recipes.shape[0]))


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
        new_ingredients = [string_var_compliant(i) for i in list(read_csv(DATASET_PATH / (file + '.csv')).iloc[:, 0])]
        new_ingredients = [i if i not in ingredients else 'Compound' + i for i in new_ingredients]
        ingredients += list(set(new_ingredients))
    return sorted(ingredients)


def get_ingredients_id_map(files: list[str]):
    from pandas import read_csv
    ingredients, indices = [], []
    for file in files:
        df = read_csv(DATASET_PATH / (file + '.csv'))
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
        df = read_csv(DATASET_PATH / (file + '.csv'))
        new_categories = list(set(df['Category']))
        for category in new_categories:
            new_ingredients = list(df.loc[df['Category'] == category].iloc[:, 0])
            new_ingredients = [string_var_compliant(i) for i in new_ingredients]
            if category in categories_ingredients.keys():
                ingredients = categories_ingredients[category]
                new_ingredients = [i if i in ingredients else 'Compound' + i for i in new_ingredients]
                categories_ingredients[category] = ingredients + new_ingredients
            else:
                categories_ingredients[category] = new_ingredients
    return categories_ingredients


setup(
    name='cmbp-experiments',  # Required
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
        'download_datasets': DownloadDatasets,
        'generate_users_scores': GenerateUsersScores,
        'generate_users_preferences': GenerateUsersPreferences,
        'generate_dataset': GenerateDataset,
        'build_and_train_nn': TrainNN,
        'extract_rules': ExtractRules,
        'generate_common_kb': GenerateCommonKB,
        'propose_recipes': ProposeRecipes,
    },
)
