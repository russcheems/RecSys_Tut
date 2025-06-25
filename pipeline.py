from data_loader import DataLoader
from typing import Type
import numpy as np
from sklearn.metrics import mean_absolute_error

def ndcg_at_k(user_true_dict, rec_list, k=10):
    """
    user_true_dict: {item_id: rating, ...}  # 该用户在测试集的真实评分
    rec_list: [item_id1, item_id2, ...]     # 推荐的Top-N商品ID
    """
    rel_pred = [user_true_dict.get(item_id, 0) for item_id in rec_list[:k]]
    rel_ideal = sorted(user_true_dict.values(), reverse=True)[:k]
    def dcg(rel):
        return sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rel))
    dcg_val = dcg(rel_pred)
    idcg_val = dcg(rel_ideal)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

class Pipeline:
    def __init__(self, data_path: str, model_class: Type, model_kwargs: dict = None):
        self.data_path = data_path
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.data_loader = DataLoader(data_path)
        self.model = None

    def run(self):
        # 加载和预处理数据
        self.data_loader.load_data()
        self.data_loader.preprocess_data()
        train_data, test_data = self.data_loader.split_train_test()
        user_item_matrix, _ = self.data_loader.create_matrix()
        user_mapping, item_mapping, rev_user_mapping, rev_item_mapping = self.data_loader.get_mappings()
        
        # 初始化并训练模型
        self.model = self.model_class(**self.model_kwargs)
        self.model.fit(user_item_matrix)
        
        # 评测：预测测试集中的评分
        y_true, y_pred = [], []
        user_item_true = {}
        user_item_pred = {}
        print("user_id\titem_id\ttrue_rating\tpred_rating")
        for _, row in test_data.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            true_rating = row['rating']
            pred_rating = self.model.predict(user_idx, item_idx)
            y_true.append(true_rating)
            y_pred.append(pred_rating)
            user_id = rev_user_mapping[user_idx]
            item_id = rev_item_mapping[item_idx]
            user_item_true.setdefault(user_id, {})[item_id] = true_rating
            user_item_pred.setdefault(user_id, {})[item_id] = pred_rating
            print(f"{user_id}\t{item_id}\t{true_rating:.2f}\t{pred_rating:.2f}")
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        mae = mean_absolute_error(y_true, y_pred)
        # 修正NDCG@10计算
        ndcg_scores = []
        for user_id in user_item_true:
            # 推荐Top-10商品ID
            user_idx = user_mapping[user_id]
            rec_item_indices = self.model.recommend(user_idx, top_k=10)
            rec_item_ids = [rev_item_mapping[i] for i in rec_item_indices]
            ndcg_scores.append(ndcg_at_k(user_item_true[user_id], rec_item_ids, k=10))
        ndcg10 = np.mean(ndcg_scores)
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test NDCG@10: {ndcg10:.4f}")
        return rmse, mae, ndcg10

    def recommend_for_user(self, user_id: str, top_k: int = 10):
        user_mapping, _, rev_user_mapping, rev_item_mapping = self.data_loader.get_mappings()
        if user_id not in user_mapping:
            raise ValueError(f"User {user_id} not found.")
        user_idx = user_mapping[user_id]
        item_indices = self.model.recommend(user_idx, top_k)
        return [rev_item_mapping[i] for i in item_indices] 