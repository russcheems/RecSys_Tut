import numpy as np
from .base import RecommenderBase

class MatrixFactorization(RecommenderBase):

    def __init__(self, n_factors=100, n_epochs=20, lr=0.005, reg=0.02):
        

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
    
    def fit(self, user_item_matrix: np.ndarray):

        self.user_item_matrix = user_item_matrix
        n_users, n_items = user_item_matrix.shape
        
        # 计算全局平均评分
        mask = user_item_matrix > 0
        self.global_mean = np.sum(user_item_matrix) / np.sum(mask)
        
        # 初始化用户和物品隐因子矩阵
        # 使用较小的随机值初始化
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # 使用随机梯度下降优化
        for epoch in range(self.n_epochs):
            # 遍历所有非零评分
            for u in range(n_users):
                for i in range(n_items):
                    if user_item_matrix[u, i] > 0:
                        # 计算预测评分
                        pred = self.global_mean + np.dot(self.user_factors[u], self.item_factors[i])
                        # 计算误差
                        error = user_item_matrix[u, i] - pred
                        
                        # 更新用户和物品隐因子
                        self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                        self.item_factors[i] += self.lr * (error * self.user_factors[u] - self.reg * self.item_factors[i])
    
    def predict(self, user: int, item: int) -> float:
        """
        预测用户对物品的评分
        
        参数:
            user (int): 用户ID
            item (int): 物品ID
            
        返回:
            float: 预测评分
        """
        if self.user_factors is None or self.item_factors is None:
            return 0.0 if self.global_mean is None else float(self.global_mean)
        
        # 预测评分 = 全局平均评分 + 用户和物品隐因子的点积
        global_mean_value = 0.0 if self.global_mean is None else float(self.global_mean)
        pred = global_mean_value + np.dot(self.user_factors[user], self.item_factors[item])
        return float(pred)
    
    def recommend(self, user: int, top_k: int = 10) -> list:

        # 检查模型是否已训练
        if self.user_item_matrix is None:
            return []
            
        # 获取用户未评分的物品
        user_ratings = self.user_item_matrix[user]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # 如果没有未评分的物品，返回空列表
        if len(unrated_items) == 0:
            return []
        
        # 预测用户对未评分物品的评分
        scores = [self.predict(user, item) for item in unrated_items]
        
        # 选择评分最高的top_k个物品
        top_indices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
        return [unrated_items[i] for i in top_indices]