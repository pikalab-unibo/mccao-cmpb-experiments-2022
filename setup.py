import sys
import random
import distutils.cmd

from utils import *
from os import system
from numpy import arange
from psyke import Extractor
from distutils.core import setup
from setuptools import find_packages
from psyke.utils.logic import pretty_clause
from pandas import Series, read_csv, DataFrame
from resources.profiles import NUTRITION_USERS
from resources.models import PATH as MODEL_PATH
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model
from resources.preferences import PATH as PREFERENCES_PATH
from tensorflow.python.framework.random_seed import set_seed
from resources.prescriptions import PATH as PRESCRIPTIONS_PATH
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog, text_to_prolog


SEED = 0
T = 1
P = 0.05
FIRST_LAYER = 16
HIDDEN_LAYER = 8
OUTPUT_LAYER = 1
EPOCHS = 10
BATCH = 32
RECIPES_LIST_FILE = '01_Recipe_Details.csv'
INGREDIENTS_FILE = '02_Ingredients.csv'
COMPOUND_INGREDIENTS_FILE = '03_Compound_Ingredients.csv'
RECIPES_FILE = '04_Recipe-Ingredients_Aliases.csv'
USERS_STYLES = ['', 'vegetarian', 'sportive', 'unhealthy']
PRESCRIPTIONS = ['day1-lunch', 'day1-dinner', 'day2-lunch', 'day2-dinner', 'day2-lunch', 'day2-dinner']
USER_AND_STYLE_OPTIONS = [
    ('user=', 'u', 'user\'s nutrition profile ([1]/2/3)'),
    ('style=', 's', 'user\'s nutrition style ([1]/2/3/4)\n\t1 = no style\n\t2 = vegan\n\t3 = sportive\n\t4 = unhealthy')
]
PRESCRIPTION_OPTIONS = [
    ('prescription=', 'p', 'prescription to satisfy ([1]/2/3/4/5/6)\n\t1 = day 1 lunch\n\t2 = day 1 dinner'
                           '\n\t3 = day 2 lunch\n\t4 = day 2 dinner\n\t5 = day 3 lunch\n\t6 = day 3 dinner')
]


class DownloadDatasets(distutils.cmd.Command):
    """
    First command to run.
    It downloads a dataset of 4 files from cosylab.iiitd.edu.in:
      - 01_Recipe_Details.csv               -> Recipe ID, Title, Source, Cuisine;
      - 02_Ingredients.csv                  -> Aliased Ingredient Name, Ingredient Synonyms, Entity ID, Category;
      - 03_Compound_Ingredients.csv         -> Compound Ingredient Name, Compound Ingredient Synonyms, entity_id,
                                               Contituent Ingredient, Category; (yes, there is a misspelling)
      - 04_Recipe-Ingredients_Aliases.csv   -> Recipe ID, Original Ingredient Name, Aliased Ingredient Name, Entity ID.
    Files are stored in resources/dataset.
    """
    description = 'download the datasets for the experiments'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        system('curl -o culinary_db.zip --url https://cosylab.iiitd.edu.in/culinarydb/static/data/CulinaryDB.zip')
        system('unzip culinary_db.zip -d ' + str(DATASET_PATH))
        system('rm -rfd culinary_db.zip')


class GenerateUsersPreferences(distutils.cmd.Command):
    """
    Second command to run.
    It generates a synthetic dataset about a single user's preferences for some ingredients and categories.
    To each ingredient is associated a uniform random value between an interval based on the previous command output.
    Values are in range from -1 (dislike) to 1 (like).
    File is stored in resources/dataset.
    """
    description = 'generate a csv file with the user\'s preferences'
    user_options = USER_AND_STYLE_OPTIONS
    min_n, max_n = -1., 1.
    min_v, max_v = 1., 10.

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:

        def compare_series_with_var_string(items: Series, string: str) -> Series:
            return Series([string_var_compliant(item) == string for item in items])

        ingredients = read_csv(DATASET_PATH / INGREDIENTS_FILE)
        compound_ingredients = read_csv(DATASET_PATH / COMPOUND_INGREDIENTS_FILE)
        scores = {}
        np.random.seed(SEED)
        for key, (min_limit, max_limit) in NUTRITION_USERS[self.user]['category_preferences'].items():
            bool_filter = compare_series_with_var_string(ingredients['Category'], key)
            local_ingredients = list(ingredients.loc[bool_filter].iloc[:, 0])
            bool_filter = compare_series_with_var_string(compound_ingredients['Category'], key)
            if any(bool_filter):
                local_ingredients += list(compound_ingredients.loc[bool_filter].iloc[:, 0])
            for ingredient in local_ingredients:
                score = np.random.uniform(low=min_limit, high=max_limit)
                if string_var_compliant(ingredient) in scores.keys():
                    scores[string_var_compliant('Compound' + ingredient)] = score
                else:
                    scores[string_var_compliant(ingredient)] = score
        for key, (min_limit, max_limit) in NUTRITION_USERS[self.user]['ingredient_preferences'].items():
            scores[string_var_compliant(key)] = np.random.uniform(low=min_limit, high=max_limit)
        scores = dict(sorted(scores.items()))
        df = DataFrame.from_records([scores])
        df = (df - self.min_v) * (self.max_n - self.min_n) / (self.max_v - self.min_v) + self.min_n
        df.to_csv(DATASET_PATH / (self.user + self.style + '.csv'), index=False)


class GenerateUsersScores(distutils.cmd.Command):
    """
    Third command to run.
    It generates a synthetic dataset about a single user's preferences for all ingredients.
    To each ingredient/category listed in a configuration file is associated a uniform random value between an interval.
    Values are in range from -1 (dislike) to 1 (like).
    File is stored in resources/dataset.
    """
    description = 'generate a file with the user\'s scores'
    user_options = USER_AND_STYLE_OPTIONS
    default_output_file_name: str = 'user_scores'
    default_hints_file: str = 'user_preferences'
    default_min: float = -1.
    default_max: float = 1.

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1
        self.min = self.default_min
        self.max = self.default_max

    def finalize_options(self) -> None:
        self.min = float(self.min)
        self.max = float(self.max)
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:
        ingredients = get_ingredients([INGREDIENTS_FILE, COMPOUND_INGREDIENTS_FILE])
        users_preferences = read_csv(DATASET_PATH / (self.user + self.style + '.csv'))
        ingredients_sublist = users_preferences.columns
        ingredients_sublist = sorted([string_var_compliant(ingredient) for ingredient in ingredients_sublist])
        ingredients_scores = DataFrame(0., index=arange(len(users_preferences.index)), columns=ingredients)
        for j, user_preferences in users_preferences.iterrows():
            for i, score in enumerate(user_preferences):
                if string_var_compliant(ingredients_sublist[i]) in ingredients:
                    ingredients_scores.at[j, ingredients_sublist[i]] = score
        ingredients_scores.to_csv(DATASET_PATH / (self.user + self.style + '.csv'), index=False)


class GenerateDataset(distutils.cmd.Command):
    """
    Fourth command to run.
    It generates a synthetic dataset about a single user's preferences for recipes.
    This dataset is the one used to train a ML model, i.e. a neural network, to predict user's preferences.
    Each row contains a recipe with a boolean value for all ingredients in the domain, plus an additional boolean value
    for the class (0 dislike, 1 like).
    File is stored in resources/dataset.
    """
    description = 'generate the labeled dataset for a specific user'
    user_options = USER_AND_STYLE_OPTIONS

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:
        from pandas import read_csv, DataFrame

        random.seed(SEED)
        ingredients_list = get_ingredients([INGREDIENTS_FILE, COMPOUND_INGREDIENTS_FILE])
        ingredients = get_ingredients_id_map([INGREDIENTS_FILE, COMPOUND_INGREDIENTS_FILE])
        recipes = read_csv(DATASET_PATH / RECIPES_FILE)
        matrix_recipes_ingredients = np.zeros(shape=(len(set(recipes.iloc[:, 0])), len(ingredients_list)))
        for i, recipe in enumerate(set(recipes.iloc[:, 0])):
            local_ingredients = list(recipes.loc[recipes['Recipe ID'] == recipe].iloc[:, -1])
            local_ingredients = [string_var_compliant(ingredients[i]) for i in local_ingredients if i in ingredients.keys()]
            for ingredient in local_ingredients:
                matrix_recipes_ingredients[i, ingredients_list.index(ingredient)] = 1
        recipes = DataFrame(matrix_recipes_ingredients, columns=ingredients_list)
        users = read_csv(DATASET_PATH / (self.user + self.style + '.csv'))
        rxu = np.dot(recipes, users.T)
        random_noise = DataFrame([random.random() > 1 - P for _ in range(rxu.shape[0])])
        labels = ((rxu > rxu.mean()) | ((rxu >= T) & random_noise)).astype(int)
        dataset = recipes.join(labels)
        dataset.columns = list(recipes.columns) + list(['target', ])
        dataset.to_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv'), index=False)


class TrainNN(distutils.cmd.Command):
    """
    Fifth command to run.
    It generates and train a neural network upon the previous generated dataset.
    At the end of the training the model is able to say if a recipe will be liked by the user or not with high accuracy.
    The model is stored in resources/models.
    """
    description = 'create and train a NN on the provided dataset'
    user_options = USER_AND_STYLE_OPTIONS
    neurons_per_layer = [FIRST_LAYER, HIDDEN_LAYER, OUTPUT_LAYER]

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:
        set_seed(SEED)
        dataset = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=2/3, random_state=SEED, stratify=dataset.iloc[:, -1])
        input_size = dataset.shape[1] - 1
        network = create_nn(input_size, self.neurons_per_layer)
        network.summary()
        network.fit(train.iloc[:, :-1], train.iloc[:, -1:], epochs=EPOCHS, batch_size=BATCH)
        loss, accuracy = network.evaluate(test.iloc[:, :-1], test.iloc[:, -1:])
        print('Accuracy on test set: ' + str(accuracy))
        network.save(MODEL_PATH / ('network_' + self.user + self.style + '.h5'))


class ExtractRules(distutils.cmd.Command):
    """
    Sixth command to run.
    It generates symbolic logic preferences that describe the internal decision-making behaviour of the trained model.
    Therefore, preferences describe the user's food preferences.
    The file is stored in resources/preferences.
    """
    description = 'extract logic preferences that describe the behaviour of the NN, i.e., the user\'s preferences'
    user_options = USER_AND_STYLE_OPTIONS
    simplify = False
    mapping = {'negative': 0, 'positive': 1}
    inverse_mapping = {0: 'negative', 1: 'positive'}

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:
        set_seed(SEED)
        dataset = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=2/3, random_state=SEED, stratify=dataset.iloc[:, -1])
        train['target'] = train['target'].apply(lambda x: 'positive' if x == 1 else 'negative')
        network = load_model(MODEL_PATH / ('network_' + self.user + self.style + '.h5'))
        network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        extractor = Extractor.cart(network, simplify=self.simplify, max_depth=10, max_leaves=30)
        theory = extractor.extract(train, self.mapping)
        with open(PREFERENCES_PATH / (self.user + self.style + '.csv'), 'w') as file:
            for rule in theory.clauses:
                file.write(pretty_clause(rule) + '.\n')


class GenerateCommonKB(distutils.cmd.Command):
    """
    Seventh command to run.
    It generates common knowledge base describing how ingredients and categories are related.
    The file is stored in resources/prescriptions.
    """
    description = 'create the predicates for food categories'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        kb = ''
        ingredients = get_ingredients([INGREDIENTS_FILE, COMPOUND_INGREDIENTS_FILE])
        categories_ingredients = get_categories_ingredients_map([INGREDIENTS_FILE, COMPOUND_INGREDIENTS_FILE])
        for k, vs in categories_ingredients.items():
            k = string_var_compliant(k)
            k = k[0].lower() + k[1:]
            for v in vs:
                kb += k + '(' + ', '.join(ingredients) + ') :-\n\t' + v + ' > 0.5.\n'
            kb += '\n'
        with open(PRESCRIPTIONS_PATH / 'kb.csv', "w") as file:
            file.write(kb)


class ProposeRecipes(distutils.cmd.Command):
    """
    Eightieth command to run.
    It generates recipes to recommend to the user. Recipes are compliant to both user's preferences and prescriptions.
    """
    description = 'for the moment just print the number of recipes satisfying both user preferences and prescriptions'
    user_options = USER_AND_STYLE_OPTIONS + PRESCRIPTION_OPTIONS
    kb = 'kb.csv'

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1
        self.prescription = PRESCRIPTIONS[0] + '.csv'

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]

    def run(self) -> None:
        from pandas import read_csv

        recipes = read_csv(DATASET_PATH / RECIPES_LIST_FILE)
        recipes_with_ingredients = list(set(read_csv(DATASET_PATH / RECIPES_FILE).iloc[:, 0]))
        data = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int).iloc[:, :-1]
        user_preferences_theory = file_to_prolog(PREFERENCES_PATH / (self.user + self.style + '.csv'))
        sys.setrecursionlimit(2000)  # because the number of ingredients is greater than 1000.

        with open(PRESCRIPTIONS_PATH / self.kb, 'r') as file:
            kb = file.read()
        with open(PRESCRIPTIONS_PATH / self.prescription, 'r') as file:
            prescriptions = file.read()
        prescriptions = text_to_prolog(kb + prescriptions)
        prescriptions_formulae = prolog_to_datalog(prescriptions)
        prescriptions_filters = formulae_to_callable(prescriptions_formulae)

        user_preferences_formulae = prolog_to_datalog(user_preferences_theory)
        user_preferences_formulae = [f for f in user_preferences_formulae if f.lhs.arg.last.name == 'positive']
        preferences_filters = formulae_to_callable(user_preferences_formulae)

        preferred_recipes = data[data.apply(preferences_filters, axis=1)]
        prescriptions_recipes = data[data.apply(prescriptions_filters, axis=1)]
        preferred_and_prescribed_recipes = prescriptions_recipes[prescriptions_recipes.apply(preferences_filters, axis=1)]

        print("\nRecipes accepted according to user's preferences: " + str(preferred_recipes.shape[0]))
        print("Recipes compliant with prescriptions: " + str(prescriptions_recipes.shape[0]))
        print("Recipes compliant to both prescriptions and user's preferences: " + str(preferred_and_prescribed_recipes.shape[0]))

        map_id_num_ing = {k: v for k, v in zip(recipes_with_ingredients, data.T.sum())}
        proposed_recipes_id = [recipes_with_ingredients[i] for i in preferred_and_prescribed_recipes.index]
        titles = recipes.loc[recipes['Recipe ID'].isin(proposed_recipes_id)].iloc[:, :2]
        titles['NumIngredients'] = [map_id_num_ing[i] for i in titles['Recipe ID']]
        titles = titles.sort_values('NumIngredients', ascending=True).iloc[:10, :]
        pd.set_option('display.max_columns', 10)
        print('\n\nBest top recipes (compliant with preferences and prescriptions with minimum amount of ingredients)')
        print(titles)


setup(
    name='cmbp-experiments',  # Required
    description='Integrating SKE into a food recommendation system, experiments.',
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
