from abc import ABC, abstractmethod
import numpy as np

class RecommenderBase(ABC):

    @abstractmethod
    def fit(self, user_item_matrix: np.ndarray):

        pass

    @abstractmethod
    def predict(self, user: int, item: int) -> float:

        pass

    @abstractmethod
    def recommend(self, user: int, top_k: int = 10) -> list:

        pass 