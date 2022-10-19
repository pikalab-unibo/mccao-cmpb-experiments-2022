## mccao-cmpb-experiments-2022

### Description
This project contains the code to run the experiments presented in the corresponding paper submitted in the journal Computer Methods and Programs in Biomedicine.
All resources, except prescriptions and user profiles that are already included, must be downloaded or generated via commands.

### Setup
To properly run the experiments you should:
1. download this repository ```git clone https://github.com/pikalab-unibo/mccao-cmpb-experiments-2022.git```
2. configure a virgin python environment ```python -m venv env_name```
3. enter into the new environment ```. env_name/bin/activate```
4. install dependencies ```pip install requirements.txt```

### Experiments

Run the following commands:
1. Download datasets: ```python -m setup download_datasets```
2. Generate user's preferences: ```python -m setup generate_users_preferences```
3. Generate user's scores: ```python -m setup generate_users_scores```
4. Generate dataset for training: ```python -m setup generate_dataset```
5. Build and train the ML model: ```python -m setup build_and_train_nn```
6. Extract logic rules: ```python -m setup extract_rules```
7. Generate common knowledge base: ```generate_common_kb```
8. Propose recipes: ```python -m setup propose_recipes```