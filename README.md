# mccao-cmpb-experiments-2022

To generate the rules that describe the preferences of a specific user run the following commands:
1. Generate user's preferences: ```python -m setup generate_users_preferences```
2. Generate user's scores: ```python -m setup generate_users_scores```
3. Generate dataset: ```python -m setup generate_dataset```
4. build and train nn: ```python -m setup build_and_train_nn```
5. extract rules: ```python -m setup extract_rules```