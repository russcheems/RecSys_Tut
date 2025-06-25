import numpy as np
from .base import RecommenderBase
from sklearn.metrics.pairwise import cosine_similarity

class UserCF(RecommenderBase):
    def __init__(self, k=20):
        self.k = k  # 近邻数
        self.user_item_matrix = None
        self.sim_matrix = None

    def fit(self, user_item_matrix: np.ndarray):
        self.user_item_matrix = user_item_matrix
        # 计算用户相似度矩阵
        self.sim_matrix = cosine_similarity(np.nan_to_num(user_item_matrix))

    def predict(self, user: int, item: int) -> float:
        # 找到对该item有评分的用户
        item_ratings = self.user_item_matrix[:, item]
        rated_users = np.where(item_ratings > 0)[0]
        if len(rated_users) == 0:
            return 0.0
        # 取与目标用户最相似的k个用户
        sim_scores = self.sim_matrix[user, rated_users]
        top_k_idx = np.argsort(sim_scores)[-self.k:]
        top_users = rated_users[top_k_idx]
        top_sims = sim_scores[top_k_idx]
        top_ratings = item_ratings[top_users]
        if np.sum(top_sims) == 0:
            return 0.0
        return np.dot(top_sims, top_ratings) / np.sum(top_sims)

    def recommend(self, user: int, top_k: int = 10) -> list:
        # 推荐用户未评分的item
        user_ratings = self.user_item_matrix[user]
        unrated_items = np.where(user_ratings == 0)[0]
        scores = [self.predict(user, item) for item in unrated_items]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [unrated_items[i] for i in top_indices] 