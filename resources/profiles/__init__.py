from pathlib import Path

PATH = Path(__file__).parents[0]

NUTRITION_STYLES = dict(
    vegan=dict(
        category_preferences=dict(
            Vegetable=(8, 10),
            Fruit=(8, 10),
            Herb=(7, 10),
            Plant=(7, 9),
            NutsSeed=(6, 9),
            Meat=(1, 1),
            Fish=(1, 2),
            Seafood=(2, 4)),
        ingredient_preferences=dict(),
    ),
    sportive=dict(
        category_preferences=dict(
            Meat=(7, 10),
            Seafood=(8, 10),
            NutsSeed=(8, 10),
            EssentialOil=(1, 3),
            BeverageAlcoholic=(1, 3),
            Bakery=(5, 7)),
        ingredient_preferences=dict(
            milk=(6, 10),
            egg=(6, 10),
        )
    ),
    unhealthy=dict(
        category_preferences=dict(
            BeverageAlcoholic=(7, 10),
            EssentialOil=(6, 10),
            Vegetable=(1, 6),
            Fruit=(2, 7),
            Seafood=(3, 6)),
        ingredient_preferences=dict(),
    ),
)

NUTRITION_USERS = dict(
    user_1=dict(
        ingredient_preferences=dict(
            Pastry=(7, 10),
            Meat=(6, 10),
            Milk=(8, 10),
            Egg=(7, 10),),
        category_preferences=dict(
            Spice=(2, 6),
            Seafood=(1, 7),
            EssentialOil=(5, 8),
            NutsSeed=(8, 10),
            BeverageAlcoholic=(1, 2),
            Fruit=(5, 10),),
        ),
    user_2=dict(
        ingredient_preferences=dict(
            Milk=(1, 3),),
        category_preferences=dict(
            Meat=(7, 10),
            Bakery=(5, 10),
            Herb=(1, 5),
            Beverage=(5, 10),
            Fruit=(7, 10),
            Spice=(1, 2),),
    ),
    user_3=dict(
        ingredient_preferences=dict(
            Milk=(6, 10),),
        category_preferences=dict(
            Meat=(7, 10),
            Bakery=(6, 10),
            Herb=(3, 6),
            Fruit=(7, 10),
            BeverageAlcoholic=(8, 10),
            Spice=(4, 7),
            Seafood=(2, 5),),
    ),
)
