from data_loader import DataLoader
from typing import Type
import numpy as np
from sklearn.metrics import mean_absolute_error

def ndcg_at_k(user_true_dict, rec_list, k=10):


    if user_true_dict is None or not isinstance(user_true_dict, dict) or len(user_true_dict) == 0:
        return 0.0
    
    if rec_list is None or not isinstance(rec_list, list) or len(rec_list) == 0:
        return 0.0
    
    # 确保k是正整数
    k = max(1, int(k))
    
    # 截取推荐列表前k个
    rec_list_k = rec_list[:k] if len(rec_list) >= k else rec_list
    
    # 获取推荐列表中每个商品的相关性得分
    rel_pred = [float(user_true_dict.get(item_id, 0)) for item_id in rec_list_k]
    
    # 获取理想排序的相关性得分
    try:
        rel_ideal = sorted([float(r) for r in user_true_dict.values()], reverse=True)[:k]
    except (TypeError, ValueError):
        # 处理无法转换为浮点数的情况
        return 0.0
    
    # 计算DCG
    def dcg(rel):
        return sum((2**float(r) - 1) / np.log2(i+2) for i, r in enumerate(rel))
    
    # 计算实际DCG和理想DCG
    try:
        dcg_val = dcg(rel_pred)
        idcg_val = dcg(rel_ideal)
        # 计算NDCG
        return float(dcg_val / idcg_val) if idcg_val > 0 else 0.0
    except Exception as e:
        print(f"计算NDCG时出错: {str(e)}")
        return 0.0

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
        
        # 获取映射
        mappings = self.data_loader.get_mappings()
        if mappings is None:
            raise ValueError("无法获取用户和商品映射，请先加载数据。")
            
        user_mapping, item_mapping, rev_user_mapping, rev_item_mapping = mappings
        
        # 初始化并训练模型
        self.model = self.model_class(**self.model_kwargs or {})
        self.model.fit(user_item_matrix)
        
        # 评测：预测测试集中的评分
        y_true, y_pred = [], []
        user_item_true = {}
        user_item_pred = {}
        print("user_id\titem_id\ttrue_rating\tpred_rating")
        
        for _, row in test_data.iterrows():
            try:
                user_idx = int(row['user_idx'])
                item_idx = int(row['item_idx'])
                true_rating = float(row['rating'])
                
                # 预测评分
                try:
                    pred_rating = float(self.model.predict(user_idx, item_idx) or 0.0)
                except Exception as e:
                    print(f"预测评分时出错: {str(e)}")
                    pred_rating = 0.0
                
                y_true.append(true_rating)
                y_pred.append(pred_rating)
                
                # 获取用户ID和商品ID
                if user_idx in rev_user_mapping and item_idx in rev_item_mapping:
                    user_id = rev_user_mapping[user_idx]
                    item_id = rev_item_mapping[item_idx]
                    user_item_true.setdefault(user_id, {})[item_id] = true_rating
                    user_item_pred.setdefault(user_id, {})[item_id] = pred_rating
                    print(f"{user_id}\t{item_id}\t{true_rating:.2f}\t{pred_rating:.2f}")
            except Exception as e:
                print(f"处理测试数据行时出错: {str(e)}")
                continue
        
        # 计算RMSE和MAE
        if len(y_true) > 0 and len(y_pred) > 0:
            rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
            mae = mean_absolute_error(y_true, y_pred)
        else:
            print("警告: 没有有效的预测结果，无法计算RMSE和MAE")
            rmse = mae = 0.0
        
        # 计算NDCG@10
        ndcg_scores = []
        for user_id in user_item_true:
            try:
                # 获取用户索引
                if user_id not in user_mapping:
                    continue
                    
                user_idx = user_mapping[user_id]
                
                # 获取推荐结果
                try:
                    rec_item_indices = self.model.recommend(user_idx, top_k=10)
                    # 确保推荐结果不为None
                    if rec_item_indices is None:
                        rec_item_indices = []
                except Exception as e:
                    print(f"为用户 {user_id} 推荐商品时出错: {str(e)}")
                    rec_item_indices = []
                
                # 转换为商品ID
                rec_item_ids = [rev_item_mapping[i] for i in rec_item_indices if i in rev_item_mapping]
                
                # 计算NDCG
                ndcg = ndcg_at_k(user_item_true[user_id], rec_item_ids, k=10)
                ndcg_scores.append(ndcg)
            except Exception as e:
                print(f"计算用户 {user_id} 的NDCG时出错: {str(e)}")
                continue
        
        # 计算平均NDCG
        ndcg10 = np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0
        
        # 打印评估结果
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test NDCG@10: {ndcg10:.4f}")
        
        return rmse, mae, ndcg10

    def recommend_for_user(self, user_id: str, top_k: int = 10):
        """
        为指定用户推荐商品
        
        参数:
            user_id (str): 用户ID
            top_k (int): 推荐商品数量
            
        返回:
            list: 推荐的商品ID列表
        """
        # 检查模型是否已初始化
        if self.model is None:
            self.data_loader.load_data()
            self.data_loader.preprocess_data()
            train_data, test_data = self.data_loader.split_train_test()
            user_item_matrix, _ = self.data_loader.create_matrix()
            self.model = self.model_class(**self.model_kwargs or {})
            self.model.fit(user_item_matrix)
        
        # 获取映射
        mappings = self.data_loader.get_mappings()
        if mappings is None:
            raise ValueError("无法获取用户和商品映射，请先加载数据。")
            
        user_mapping, _, rev_user_mapping, rev_item_mapping = mappings
        
        # 检查用户是否存在
        if user_mapping is None or user_id not in user_mapping:
            raise ValueError(f"用户 {user_id} 不存在或映射未初始化。")
            
        user_idx = user_mapping[user_id]
        
        # 获取推荐结果
        try:
            item_indices = self.model.recommend(user_idx, top_k)
            # 确保返回的是列表且不为None
            if item_indices is None:
                return []
            # 转换为商品ID
            return [rev_item_mapping[i] for i in item_indices if i in rev_item_mapping]
        except Exception as e:
            print(f"推荐过程中出错: {str(e)}")
            return []
        
    def print_user_recommendations(self, user_id: str, top_k: int = 10):
        """
        打印用户预测喜欢的前k个商品和实际购买评分前k个商品
        
        参数:
            user_id (str): 用户ID
            top_k (int): 推荐商品数量
        """
        # 加载数据
        if self.model is None:
            self.data_loader.load_data()
            self.data_loader.preprocess_data()
            train_data, test_data = self.data_loader.split_train_test()
            user_item_matrix, _ = self.data_loader.create_matrix()
            self.model = self.model_class(**self.model_kwargs or {})
            self.model.fit(user_item_matrix)
            
        # 获取映射
        mappings = self.data_loader.get_mappings()
        if mappings is None:
            raise ValueError("无法获取用户和商品映射，请先加载数据。")
            
        user_mapping, item_mapping, rev_user_mapping, rev_item_mapping = mappings
        
        # 检查用户是否存在
        if user_mapping is None or user_id not in user_mapping:
            raise ValueError(f"用户 {user_id} 不存在或映射未初始化。")
            
        user_idx = user_mapping[user_id]
        
        # 获取测试数据
        test_data = self.data_loader.test_data
        if test_data is None:
            raise ValueError("测试数据未加载，请先运行pipeline.run()方法。")
        
        # 过滤出用户的测试数据
        if 'userID' not in test_data.columns:
            raise ValueError("测试数据中没有userID列。")
            
        user_test_data = test_data[test_data['userID'] == user_id]
        
        # 如果用户在测试集中没有评分，则只显示推荐结果
        if len(user_test_data) == 0:
            print(f"用户 {user_id} 在测试集中没有评分记录")
            rec_items = self.recommend_for_user(user_id, top_k)
            print(f"\n预测用户 {user_id} 喜欢的前 {top_k} 个商品:")
            for i, item_id in enumerate(rec_items):
                print(f"{i+1}. 商品ID: {item_id}")
            return
        
        # 构建用户实际评分字典
        user_ratings = {}
        for _, row in user_test_data.iterrows():
            if 'itemID' not in row or 'rating' not in row:
                continue
            item_id = row['itemID']
            rating = row['rating']
            user_ratings[item_id] = float(rating)
        
        # 如果没有有效评分，提前返回
        if not user_ratings:
            print(f"用户 {user_id} 在测试集中没有有效的评分记录")
            return
        
        # 获取用户实际评分最高的前k个商品
        top_rated_items = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 获取推荐的前k个商品
        rec_items = self.recommend_for_user(user_id, top_k)
        
        # 打印结果
        print(f"\n用户 {user_id} 实际评分最高的前 {top_k} 个商品:")
        for i, (item_id, rating) in enumerate(top_rated_items):
            print(f"{i+1}. 商品ID: {item_id}, 评分: {rating:.2f}")
        
        print(f"\n预测用户 {user_id} 喜欢的前 {top_k} 个商品:")
        for i, item_id in enumerate(rec_items):
            # 如果推荐的商品在用户实际评分中，显示实际评分
            actual_rating = user_ratings.get(item_id, "未评分")
            if isinstance(actual_rating, (int, float)):
                print(f"{i+1}. 商品ID: {item_id}, 实际评分: {actual_rating:.2f}")
            else:
                print(f"{i+1}. 商品ID: {item_id}, 实际评分: {actual_rating}")
        
        # 计算并打印NDCG@k
        ndcg = ndcg_at_k(user_ratings, rec_items, k=top_k)
        print(f"\nNDCG@{top_k}: {ndcg:.4f}")