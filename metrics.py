import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RecommenderMetrics:
    """
    推荐系统评估指标类
    包含各种评估推荐系统性能的指标
    """
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        """
        计算Top-K推荐的准确率
        
        参数:
            recommended_items (list): 推荐的商品列表
            relevant_items (list): 相关（用户实际喜欢的）商品列表
            k (int): 推荐列表长度
            
        返回:
            float: 准确率
        """
        # 确保推荐列表长度为k
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
        
        # 计算命中数量
        hits = len(set(recommended_items) & set(relevant_items))
        
        # 计算准确率
        precision = hits / min(k, len(recommended_items)) if len(recommended_items) > 0 else 0
        
        return precision
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        """
        计算Top-K推荐的召回率
        
        参数:
            recommended_items (list): 推荐的商品列表
            relevant_items (list): 相关（用户实际喜欢的）商品列表
            k (int): 推荐列表长度
            
        返回:
            float: 召回率
        """
        # 确保推荐列表长度为k
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
        
        # 计算命中数量
        hits = len(set(recommended_items) & set(relevant_items))
        
        # 计算召回率
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
        
        return recall
    
    @staticmethod
    def f1_at_k(recommended_items, relevant_items, k=10):
        """
        计算Top-K推荐的F1分数
        
        参数:
            recommended_items (list): 推荐的商品列表
            relevant_items (list): 相关（用户实际喜欢的）商品列表
            k (int): 推荐列表长度
            
        返回:
            float: F1分数
        """
        precision = RecommenderMetrics.precision_at_k(recommended_items, relevant_items, k)
        recall = RecommenderMetrics.recall_at_k(recommended_items, relevant_items, k)
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    @staticmethod
    def mean_absolute_error(predictions, actual):
        """
        计算平均绝对误差
        
        参数:
            predictions (list): 预测评分列表
            actual (list): 实际评分列表
            
        返回:
            float: 平均绝对误差
        """
        return mean_absolute_error(actual, predictions)
    
    @staticmethod
    def root_mean_squared_error(predictions, actual):
        """
        计算均方根误差
        
        参数:
            predictions (list): 预测评分列表
            actual (list): 实际评分列表
            
        返回:
            float: 均方根误差
        """
        return np.sqrt(mean_squared_error(actual, predictions))