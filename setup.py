import sys
import random
import os.path
import distutils.cmd

from utils import *
from os import system
from numpy import arange
from statistics import mean
from psyke import Extractor
from distutils.core import setup
from setuptools import find_packages
from psyke.utils.logic import pretty_clause
from pandas import Series, read_csv, DataFrame
from resources.models import PATH as MODEL_PATH
from resources.results import PATH as RESULTS_PATH
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model
from resources.preferences import PATH as PREFERENCES_PATH
from tensorflow.python.framework.random_seed import set_seed
from resources.prescriptions import PATH as PRESCRIPTIONS_PATH
from resources.profiles import NUTRITION_USERS, NUTRITION_STYLES
from psyki.logic.datalog.grammar.adapters.tuppy import prolog_to_datalog
from psyki.logic.prolog.grammar.adapters.tuppy import file_to_prolog, text_to_prolog


SEED = 0
# Extraction hyper-parameters
MAX_PROPOSED_RECIPES_PER_PRESCRIPTION = 20
MAX_INGREDIENTS_IN_RULES = 10  # None = no limits
MAX_EXTRACTED_RULES = 50  # None = no limits
# ML hyper-parameters
TRAIN_RATIO = 0.5
FIRST_LAYER = 16
HIDDEN_LAYER = 8
OUTPUT_LAYER = 1
EPOCHS = 20
BATCH = 32
# Additional hyper-parameters
RECIPES_LIST_FILE = '01_Recipe_Details.csv'
INGREDIENTS_FILE = '02_Ingredients.csv'
COMPOUND_INGREDIENTS_FILE = '03_Compound_Ingredients.csv'
RECIPES_FILE = '04_Recipe-Ingredients_Aliases.csv'
USERS_STYLES = ['', 'vegan', 'sportive', 'unhealthy']
PRESCRIPTIONS = ['day1-lunch', 'day1-dinner', 'day2-lunch', 'day2-dinner', 'day3-lunch', 'day3-dinner']
USER_AND_STYLE_OPTIONS = [
    ('user=', 'u', 'user\'s nutrition profile ([1]/2/3)'),
    ('style=', 's', 'user\'s nutrition style ([1]/2/3/4)\n\t1 = no style\n\t2 = vegan\n\t3 = sportive\n\t4 = unhealthy')
]
PRESCRIPTION_OPTIONS = [
    ('prescription=', 'p', 'prescription to satisfy ([1]/2/3/4/5/6)\n\t1 = day 1 lunch\n\t2 = day 1 dinner'
                           '\n\t3 = day 2 lunch\n\t4 = day 2 dinner\n\t5 = day 3 lunch\n\t6 = day 3 dinner'),
]
RECIPE_OPTION = [
    ('recipe=', 'r', 'recipe identifier (1 by default)'),
]
USE_RULES = True
sys.setrecursionlimit(2000)  # because the number of ingredients is greater than 1000.


class RunAll(distutils.cmd.Command):
    """
    Run all commands in sequence for each user and prescription.
    """
    description = 'run all the experiments'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        if not os.path.isfile(DATASET_PATH / RECIPES_FILE):
            system('python -m setup download_datasets')
        if not os.path.isfile(PRESCRIPTIONS_PATH / 'kb.csv'):
            system('python -m setup generate_common_kb')

        for user in range(1, 4):
            for style in range(1, len(USERS_STYLES) + 1):
                system('python -m setup generate_users_preferences -u ' + str(user) + ' -s ' + str(style))
                system('python -m setup generate_users_scores -u ' + str(user) + ' -s ' + str(style))
                system('python -m setup generate_dataset -u ' + str(user) + ' -s ' + str(style))
                system('python -m setup build_and_train_nn -u ' + str(user) + ' -s ' + str(style))
                system('python -m setup extract_rules -u ' + str(user) + ' -s ' + str(style))
                for prescription in range(1, len(PRESCRIPTIONS) + 1):
                    system('python -m setup propose_recipes -u ' + str(user) + ' -s ' + str(style) + ' -p ' + str(prescription))


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


class GenerateCommonKB(distutils.cmd.Command):
    """
    Second command to run.
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


class GenerateUsersPreferences(distutils.cmd.Command):
    """
    Third command to run.
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

        def compute_scores(user_or_style) -> dict[str, float]:
            result = {}
            for key, (min_limit, max_limit) in user_or_style['category_preferences'].items():
                bool_filter = compare_series_with_var_string(ingredients['Category'], key)
                local_ingredients = list(ingredients.loc[bool_filter].iloc[:, 0])
                bool_filter = compare_series_with_var_string(compound_ingredients['Category'], key)
                if any(bool_filter):
                    local_ingredients += list(compound_ingredients.loc[bool_filter].iloc[:, 0])
                for ingredient in local_ingredients:
                    score = np.random.uniform(low=min_limit, high=max_limit)
                    if string_var_compliant(ingredient) in result.keys():
                        result[string_var_compliant('Compound' + ingredient)] = score
                    else:
                        result[string_var_compliant(ingredient)] = score
            for key, (min_limit, max_limit) in user_or_style['ingredient_preferences'].items():
                result[string_var_compliant(key)] = np.random.uniform(low=min_limit, high=max_limit)
            result = dict(sorted(result.items()))
            return result

        ingredients = read_csv(DATASET_PATH / INGREDIENTS_FILE)
        compound_ingredients = read_csv(DATASET_PATH / COMPOUND_INGREDIENTS_FILE)
        np.random.seed(SEED)
        style_scores = compute_scores(NUTRITION_STYLES[self.style]) if self.style != '' else {}
        scores = compute_scores(NUTRITION_USERS[self.user])
        for k, _ in style_scores.items():
            if k not in scores.keys():
                scores[k] = style_scores[k]
        df = DataFrame.from_records([scores])
        df = (df - self.min_v) * (self.max_n - self.min_n) / (self.max_v - self.min_v) + self.min_n
        df.to_csv(DATASET_PATH / (self.user + self.style + '.csv'), index=False)


class GenerateUsersScores(distutils.cmd.Command):
    """
    Fourth command to run.
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
    Fifth command to run.
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
        max_rxu = rxu.max()
        min_rxu = rxu.min()
        p = DataFrame([random.uniform(0, 1) for _ in range(rxu.shape[0])])
        x = DataFrame((rxu - min_rxu) / (max_rxu - min_rxu))
        labels = ((rxu > np.quantile(rxu, 0.75)) + (DataFrame(rxu > 0) * x > p)).astype(int)
        dataset = recipes.join(labels)
        dataset.columns = list(recipes.columns) + list(['target', ])
        ratio = dataset.loc[dataset['target'] == 1].shape[0] / dataset.shape[0]
        print('positive class ratio: ' + str(ratio))
        dataset.to_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv'), index=False)
        DataFrame([ratio]).to_csv(DATASET_PATH / ('liked_recipes_' + self.user + self.style + '.csv'), index=False, header=['liked_recipes_ratio'])


class TrainNN(distutils.cmd.Command):
    """
    Sixth command to run.
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
        train, test = train_test_split(dataset, train_size=TRAIN_RATIO, random_state=SEED, stratify=dataset.iloc[:, -1])
        input_size = dataset.shape[1] - 1
        network = create_nn(input_size, self.neurons_per_layer)
        network.summary()
        network.fit(train.iloc[:, :-1], train.iloc[:, -1:], epochs=EPOCHS, batch_size=BATCH)
        loss, accuracy, precision = network.evaluate(test.iloc[:, :-1], test.iloc[:, -1:])
        print('Accuracy on test set: ' + str(accuracy))
        print('Precision on test set: ' + str(precision))
        file_name = 'network_' + self.user + self.style
        network.save(MODEL_PATH / (file_name + '.h5'))
        DataFrame([[loss], [accuracy], [precision]]).T\
            .to_csv(MODEL_PATH / (file_name + '.csv'),
                    index_label=False, index=False, header=['loss', 'accuracy', 'precision'])


class ExtractRules(distutils.cmd.Command):
    """
    Seventh command to run.
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
        from utils import precision
        set_seed(SEED)
        dataset = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int)
        train, test = train_test_split(dataset, train_size=TRAIN_RATIO, random_state=SEED, stratify=dataset.iloc[:, -1])
        train['target'] = train['target'].apply(lambda x: 'positive' if x == 1 else 'negative')
        network = load_model(MODEL_PATH / ('network_' + self.user + self.style + '.h5'), custom_objects={'precision': precision})
        network.compile(optimizer='adam', metrics='accuracy', loss='binary_crossentropy')
        extractor = Extractor.cart(network, simplify=self.simplify, max_depth=MAX_INGREDIENTS_IN_RULES, max_leaves=MAX_EXTRACTED_RULES)
        theory = extractor.extract(train, self.mapping)
        with open(PREFERENCES_PATH / (self.user + self.style + '.csv'), 'w') as file:
            for rule in theory.clauses:
                file.write(pretty_clause(rule) + '.\n')

        model_predictions = DataFrame(network.predict(test.iloc[:, :-1]), columns=['new_target'], index=test.index)
        test = test.join(model_predictions)
        # Accuracy = (True Positive + True Negative) / (Total)
        liked_recipes = get_liked_recipes(theory, test)  # True Positive + False Positive
        true_liked_recipes = liked_recipes[liked_recipes['target'] == 1].shape[0]  # True Positive
        disliked_recipes = get_liked_recipes(theory, test, like=False)  # True Negative + False Negative
        true_disliked_recipes = disliked_recipes[disliked_recipes['target'] == 0].shape[0]  # True Negative
        accuracy = (true_liked_recipes + true_disliked_recipes) / test.shape[0]
        # precision = (True Positive) / ( True Positive + False Positive)
        false_liked_recipes = liked_recipes[liked_recipes['target'] == 0].shape[0]  # False Positive
        prec = true_liked_recipes / (true_liked_recipes + false_liked_recipes)
        # Fidelity = accuracy on predictor
        true_liked_recipes = liked_recipes[liked_recipes['new_target'] >= 0.5].shape[0]  # True Positive
        true_disliked_recipes = disliked_recipes[disliked_recipes['new_target'] < 0.5].shape[0]  # True Negative
        fidelity = (true_liked_recipes + true_disliked_recipes) / test.shape[0]
        result = DataFrame([[accuracy], [prec], [fidelity]]).T
        result.to_csv(PREFERENCES_PATH / (self.user + self.style + '_accuracy.csv'), index_label=False, index=False, header=['accuracy', 'precision', 'fidelity'])


class ProposeRecipes(distutils.cmd.Command):
    """
    Eightieth command to run.
    It generates recipes to recommend to the user. Recipes are compliant to both user's preferences and prescriptions.
    """
    description = 'Print the recipes satisfying both user preferences and prescriptions and save the ids into files.'
    user_options = USER_AND_STYLE_OPTIONS + PRESCRIPTION_OPTIONS + [('log=', 'l', '.'), ]
    kb = 'kb.csv'

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1
        self.prescription = 1
        self.log = 0

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]
        self.prescription = PRESCRIPTIONS[int(self.prescription) - 1]
        self.log = int(self.log)

    def run(self) -> None:
        prescription_file = self.prescription + '.csv'
        recipes = read_csv(DATASET_PATH / RECIPES_LIST_FILE)
        recipes_with_ingredients = list(set(read_csv(DATASET_PATH / RECIPES_FILE).iloc[:, 0]))
        data = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int)
        map_id_num_ing = {k: v for k, v in zip(recipes_with_ingredients, data.iloc[:, :-1].T.sum())}
        _, test = train_test_split(data, train_size=TRAIN_RATIO, random_state=SEED, stratify=data.iloc[:, -1])
        user_preferences_theory = file_to_prolog(PREFERENCES_PATH / (self.user + self.style + '.csv'))
        # Prescriptions
        with open(PRESCRIPTIONS_PATH / self.kb, 'r') as file:
            kb = file.read()
        with open(PRESCRIPTIONS_PATH / prescription_file, 'r') as file:
            prescriptions = file.read()
        prescriptions = text_to_prolog(kb + prescriptions)
        prescriptions_formulae = prolog_to_datalog(prescriptions)
        prescriptions_filters = formulae_to_callables(prescriptions_formulae)
        # Preferences
        if USE_RULES:
            preferred_recipes = get_liked_recipes(user_preferences_theory, test)
        else:
            model = load_model(MODEL_PATH / ('network_' + self.user + self.style + '.h5'))
            classes = model.predict(test.iloc[:, :-1])
            new_test = test.copy()
            new_test['new_target'] = [1 if x > 0.5 else 0 for x in classes]
            preferred_recipes = new_test.loc[new_test['new_target'] == 1].iloc[:, :-1]
        print("\nRecipes accepted according to user's preferences: " + str(preferred_recipes.shape[0]) + '\n\n\n')

        total_proposed_recipes, total_liked_recipes = [], []
        for i, prescription in enumerate(prescriptions_filters):
            prescribed = test[test.apply(prescription, axis=1)]
            liked_and_prescribed = preferred_recipes[preferred_recipes.apply(prescription, axis=1)].copy()
            proposed_recipes_id = [recipes_with_ingredients[i] for i in liked_and_prescribed.index]
            titles = recipes.loc[recipes['Recipe ID'].isin(proposed_recipes_id)].iloc[:, :2]
            num_ing = [map_id_num_ing[i] for i in titles['Recipe ID']]
            liked_and_prescribed['NumIngredients'] = num_ing
            liked_and_prescribed = liked_and_prescribed.sort_values('NumIngredients', ascending=True)
            liked_and_prescribed = liked_and_prescribed.iloc[:MAX_PROPOSED_RECIPES_PER_PRESCRIPTION, :]
            print("Recipes compliant with prescription " + str(i + 1) + ": " + str(prescribed.shape[0]))
            d = liked_and_prescribed.shape[0]  # Positive = True Positive + False Negative
            if d > 0:
                n = liked_and_prescribed.loc[liked_and_prescribed['target'] == 1].shape[0]  # True Positive
            else:
                n = 0
            print("Recipes compliant to both prescriptions and user's preferences: " + str(d))
            total_proposed_recipes.append(d)
            total_liked_recipes.append(n)
            if d > 0:
                print('Prescription ' + str(i + 1) + ' accuracy: ' + str(n/d))
            else:
                print('Prescription ' + str(i + 1) + ': no proposed recipes!')

            if self.log == 1:
                titles['NumIngredients'] = num_ing
                titles = titles.sort_values('NumIngredients', ascending=True)
                pd.set_option('display.max_columns', 10)
                print('\n\nBest top recipes')
                print(titles.iloc[:10, :])
                print('\n\n' + 50 * '-' + '\n\n')
                file_name = (self.user + self.style + '_' + self.prescription + '_number_' + str(i+1) + '.csv')
                titles.iloc[:, 0].to_csv(RESULTS_PATH / file_name, index_label=False, index=False)

        d = sum(total_proposed_recipes)
        n = sum(total_liked_recipes)
        total_proposed_recipes.append(d)
        total_liked_recipes.append(n)
        precisions = [n / d if d > 0 else 0 for n, d in zip(total_liked_recipes, total_proposed_recipes)]
        print('Total precision: ' + str(precisions[-1]))
        result = DataFrame([total_proposed_recipes, total_liked_recipes, precisions]).T
        if USE_RULES:
            file_name = self.user + self.style + '_' + self.prescription + '.csv'
        else:
            file_name = self.user + self.style + '_' + self.prescription + '_with_NN.csv'
        result.to_csv(RESULTS_PATH / file_name, index_label=False, index=False, header=['proposed', 'liked', 'precision'])


# Post test analysis
class ComputeStatistics(distutils.cmd.Command):
    """
    Generate latex-like tables that summarise the results.
    """
    description = 'compute statistics upon the results'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        # Generate latex code for a table summarising the accuracy of the neural networks per users
        #
        # users |   liked ratio   |   net acc   |  net prec |   rule acc   |   rule prec   |   rule fid
        # u1    |                 |             |           |              |               |
        # u2    |                 |             |           |              |               |
        # u3    |                 |             |           |              |               |
        results = 'users & liked ratio & net acc & net prec & rules acc & rules prec & rules fid\n'
        all_stats = []
        i = 1
        for u in range(1, 4):
            for s in range(1, len(USERS_STYLES) + 1):
                user = 'user_' + str(u)
                style = USERS_STYLES[int(s) - 1]
                net_data = pd.read_csv(MODEL_PATH / ('network_' + user + style + '.csv'))
                rule_data = pd.read_csv(PREFERENCES_PATH / (user + style + '_accuracy.csv'))
                class_data = pd.read_csv(DATASET_PATH / ('liked_recipes_' + user + style + '.csv'))
                net_acc, net_prec = round(net_data.iloc[-1, -2], 4), round(net_data.iloc[-1, -1], 4)
                rule_acc, rule_prec, rule_fidelity = round(rule_data.iloc[0, 0], 4), round(rule_data.iloc[0, 1], 4), round(rule_data.iloc[0, 2], 4)
                liked_ratio = round(class_data.iloc[-1, -1], 4)
                all_stats.append([liked_ratio, net_acc, net_prec, rule_acc, rule_prec, rule_fidelity])
                results += ' & '.join(['user ' + str(i), str(liked_ratio), str(net_acc), str(net_prec), str(rule_acc), str(rule_prec), str(rule_fidelity)]) + r'\\' + '\n'
                i += 1
        results += r'\hline' + '\n'
        stats = DataFrame(all_stats)
        results += 'all & ' + ' & '.join(str(round(x, 4)) for x in stats.mean(axis=0)) + r'\\' + '\n'
        with open('table_net_rule_accuracy.csv', "w") as file:
            file.write(results)

        # Generate latex code for a table summarising the precision of the proposed recipes
        #
        # users |   d1 lunch    |   d1 dinner   |   d2 lunch    |   d2 dinner   |   d3 lunch    |   d3 dinner   |   all
        # u1    |               |               |               |               |               |               |
        # u2    |               |               |               |               |               |               |
        # u3    |               |               |               |               |               |               |
        results = 'users & ' + ' & '.join(PRESCRIPTIONS) + r' & all\\' + '\n' + r'\hline\hline' + '\n'
        all_stats = []
        i = 1
        for u in range(1, 4):
            for s in range(1, len(USERS_STYLES) + 1):
                user = 'user_' + str(u)
                style = USERS_STYLES[int(s) - 1]
                results += 'user ' + str(i)
                i += 1
                partial_stats = []
                for p in range(1, len(PRESCRIPTIONS) + 1):
                    prescription = PRESCRIPTIONS[p - 1]
                    prec = pd.read_csv(RESULTS_PATH / (user + style + '_' + prescription + '.csv')).iloc[-1, -1]
                    results += ' & ' + str(round(prec, 4))
                    partial_stats.append(round(prec, 4))
                results += r' & ' + str(round(mean(partial_stats), 4)) + r'\\' + '\n'
                all_stats.append(partial_stats)
        stats = DataFrame(all_stats)
        stats = stats.mean(axis=0)
        results += 'all & ' + ' & '.join(str(round(x, 4)) for x in stats.tolist()) + ' & ' + str(round(stats.mean(), 4)) + r'\\' + '\n'
        with open('table_propose_recipes_accuracy.csv', "w") as file:
            file.write(results)


class ExplainRecommendation(distutils.cmd.Command):
    description = 'explain why the user likes or not the recipe'
    user_options = USER_AND_STYLE_OPTIONS + RECIPE_OPTION

    def initialize_options(self) -> None:
        self.user = 1
        self.style = 1
        self.recipe = 1

    def finalize_options(self) -> None:
        self.user = 'user_' + str(self.user)
        self.style = USERS_STYLES[int(self.style) - 1]
        self.recipe = int(self.recipe)

    def run(self) -> None:
        recipe_info = read_csv(DATASET_PATH / RECIPES_LIST_FILE).iloc[self.recipe - 1, :]
        recipes_with_ingredients = list(set(read_csv(DATASET_PATH / RECIPES_FILE).iloc[:, 0]))
        data = read_csv(DATASET_PATH / ('dataset_' + self.user + self.style + '.csv')).astype(int)
        recipe_id = recipes_with_ingredients.index(self.recipe)
        recipe = DataFrame(data.iloc[recipe_id, :]).T
        user_preferences_theory = file_to_prolog(PREFERENCES_PATH / (self.user + self.style + '.csv'))
        preferences_formulae = sort_formulae_by_size(prolog_to_datalog(user_preferences_theory))
        for formula in preferences_formulae:
            formula_filter = formula_to_callable(formula, {})
            result = recipe[recipe.apply(formula_filter, axis=1)]
            if not result.empty:
                string = ''
                user_opinion = result.iloc[0, -1]
                if user_opinion == 1 == recipe.iloc[0, -1]:
                    string += 'User likes recipe ' + recipe_info[1] + ' because:\n'
                elif user_opinion == 0 == recipe.iloc[0, -1]:
                    string += 'User doen\'t like recipe ' + recipe_info[1] + ' because:\n'
                else:
                    string += 'Rule fails to correctly predict the class of the recipe\n'
                    print(string)
                    break
                has, not_has = formula_to_ingredients(formula)
                string += '\t recipe has the following ingredients:\n\t\t'
                string += '\n\t\t'.join(has)
                string += '\n\t recipe hasn\'t the following ingredients:\n\t\t'
                string += '\n\t\t'.join(not_has)
                recipe_ingredients = [ingredient for ingredient, value in zip(recipe.columns[:-1], recipe.iloc[0, :-1].to_list()) if value == 1]
                string += '\nIngredients of the recipe:\n\t'
                string += '\n\t'.join(recipe_ingredients)
                print(string)
                break


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
        'run_all': RunAll,
        'download_datasets': DownloadDatasets,
        'generate_common_kb': GenerateCommonKB,
        'generate_users_preferences': GenerateUsersPreferences,
        'generate_users_scores': GenerateUsersScores,
        'generate_dataset': GenerateDataset,
        'build_and_train_nn': TrainNN,
        'extract_rules': ExtractRules,
        'propose_recipes': ProposeRecipes,
        'compute_statistics': ComputeStatistics,
        'explain_recommendation': ExplainRecommendation,
    },
)
