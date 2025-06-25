import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    负责加载和预处理推荐系统数据集
    """
    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        初始化数据加载器
        
        参数:
            file_path (str): 数据集文件路径
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
    
    def load_data(self):
        """
        加载数据集
        
        返回:
            pd.DataFrame: 加载的数据集
        """
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def preprocess_data(self):
        """
        预处理数据，创建用户和商品的映射
        """
        if self.data is None:
            self.load_data()
        
        # 创建用户和商品的映射（将ID映射为连续的整数索引）
        unique_users = self.data['userID'].unique()
        unique_items = self.data['itemID'].unique()
        
        self.user_mapping = {user: i for i, user in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        
        # 添加映射后的索引列
        self.data['user_idx'] = self.data['userID'].apply(lambda x: self.user_mapping[x])
        self.data['item_idx'] = self.data['itemID'].apply(lambda x: self.item_mapping[x])
        
        return self.data
    
    def split_train_test(self):
        """
        将数据集分割为训练集和测试集
        
        返回:
            tuple: (训练集, 测试集)
        """
        if self.data is None:
            self.load_data()
            
        if 'user_idx' not in self.data.columns:
            self.preprocess_data()
        
        self.train_data, self.test_data = train_test_split(
            self.data, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        return self.train_data, self.test_data
    
    def create_matrix(self):
        """
        创建用户-商品评分矩阵和商品-用户评分矩阵
        
        返回:
            tuple: (用户-商品矩阵, 商品-用户矩阵)
        """
        if self.train_data is None:
            self.split_train_test()
        
        if self.user_mapping is None:
            self.preprocess_data()
            
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # 创建用户-商品评分矩阵
        self.user_item_matrix = np.zeros((n_users, n_items))
        for _, row in self.train_data.iterrows():
            self.user_item_matrix[int(row['user_idx']), int(row['item_idx'])] = row['rating']
        
        # 创建商品-用户评分矩阵（转置）
        self.item_user_matrix = self.user_item_matrix.T
        
        return self.user_item_matrix, self.item_user_matrix
    
    def get_user_item_matrix(self):
        """
        获取用户-商品评分矩阵
        
        返回:
            np.ndarray: 用户-商品评分矩阵
        """
        if self.user_item_matrix is None:
            self.create_matrix()
        return self.user_item_matrix
    
    def get_item_user_matrix(self):
        """
        获取商品-用户评分矩阵
        
        返回:
            np.ndarray: 商品-用户评分矩阵
        """
        if self.item_user_matrix is None:
            self.create_matrix()
        return self.item_user_matrix
    
    def get_mappings(self):
        """
        获取ID映射
        
        返回:
            tuple: (用户映射, 商品映射, 反向用户映射, 反向商品映射)
        """
        if self.user_mapping is None:
            self.preprocess_data()
        return (
            self.user_mapping, 
            self.item_mapping, 
            self.reverse_user_mapping, 
            self.reverse_item_mapping
        )