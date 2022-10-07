from pathlib import Path
from pandas import DataFrame

PATH = Path(__file__).parents[0]


def get_feature_mapping(dataset: DataFrame) -> dict[str:int]:
    return {name: i for i, name in enumerate(dataset.columns)}
