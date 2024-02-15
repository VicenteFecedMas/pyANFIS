from .types.algorithm import Algorithm

from .types.takagi_sugeno import TakagiSugeno
from .types.tsukamoto import Tsukamoto
from .types.lee import Lee

from .consequents import Consequents

__all__ = ["Consequents", "Algorithm", "TakagiSugeno", "Tsukamoto", "Lee"]