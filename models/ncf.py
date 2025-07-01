import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import RecommenderBase

class NCF(RecommenderBase):

    def __init__(self, n_factors=8, layers=[16, 8], lr=0.001, n_epochs=20, batch_size=256, dropout=0.2, alpha=0.5):


        self.n_factors = n_factors
        self.layers = layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.alpha = alpha
        
        # 初始化模型参数
        self.user_item_matrix = None
        self.n_users = None
        self.n_items = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        
    def _init_model(self):
        """
        初始化PyTorch模型
        """
        self.model = NCFModel(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            layers=self.layers,
            dropout=self.dropout,
            alpha=self.alpha
        ).to(self.device)
    
    def _create_dataset(self):
        """
        创建训练数据集
        
        返回:
            tuple: (用户索引, 物品索引, 评分)
        """
        # 获取所有非零评分的索引和值
        user_indices, item_indices = np.where(self.user_item_matrix > 0)
        ratings = self.user_item_matrix[user_indices, item_indices]
        
        return user_indices, item_indices, ratings
    
    def fit(self, user_item_matrix: np.ndarray):
        """
        训练NCF模型
        
        参数:
            user_item_matrix (np.ndarray): 用户-物品评分矩阵
        """
        try:
            print(f"开始训练NCF模型，使用设备: {self.device}")
            print(f"模型参数: n_factors={self.n_factors}, layers={self.layers}, lr={self.lr}, n_epochs={self.n_epochs}, batch_size={self.batch_size}")
            
            self.user_item_matrix = user_item_matrix
            self.n_users, self.n_items = user_item_matrix.shape
            print(f"数据集大小: {self.n_users} 用户, {self.n_items} 物品")
            
            # 初始化模型
            self._init_model()
            
            # 创建数据集
            user_indices, item_indices, ratings = self._create_dataset()
            n_samples = len(ratings)
            print(f"训练样本数量: {n_samples}")
            
            # 转换为PyTorch张量
            try:
                users = torch.LongTensor(user_indices).to(self.device)
                items = torch.LongTensor(item_indices).to(self.device)
                ratings = torch.FloatTensor(ratings).to(self.device)
            except Exception as e:
                print(f"转换数据到张量时出错: {str(e)}")
                # 尝试使用CPU
                self.device = torch.device("cpu")
                print(f"切换到CPU设备")
                users = torch.LongTensor(user_indices).to(self.device)
                items = torch.LongTensor(item_indices).to(self.device)
                ratings = torch.FloatTensor(ratings).to(self.device)
            
            # 定义优化器和损失函数
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            # 训练模型
            self.model.train()
            print("开始训练...")
            for epoch in range(self.n_epochs):
                # 打乱数据
                indices = np.random.permutation(n_samples)
                users = users[indices]
                items = items[indices]
                ratings = ratings[indices]
                
                # 批次训练
                total_loss = 0
                n_batches = (n_samples + self.batch_size - 1) // self.batch_size
                
                for i in range(0, n_samples, self.batch_size):
                    try:
                        batch_users = users[i:i+self.batch_size]
                        batch_items = items[i:i+self.batch_size]
                        batch_ratings = ratings[i:i+self.batch_size]
                        
                        # 前向传播
                        predictions = self.model(batch_users, batch_items)
                        loss = criterion(predictions, batch_ratings)
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_ratings)
                        
                        # 显示批次进度
                        batch_idx = i // self.batch_size + 1
                        if batch_idx % 50 == 0 or batch_idx == n_batches:
                            print(f"Epoch {epoch+1}/{self.n_epochs}, Batch {batch_idx}/{n_batches}")
                            
                    except Exception as e:
                        print(f"训练批次时出错: {str(e)}")
                        continue
                
                avg_loss = total_loss / n_samples
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
            
            # 切换到评估模式
            self.model.eval()
            print("NCF模型训练完成")
            
        except Exception as e:
            print(f"训练NCF模型时出错: {str(e)}")
            # 确保模型处于评估模式
            if self.model is not None:
                self.model.eval()
    
    def predict(self, user: int, item: int) -> float:
        """
        预测用户对物品的评分
        
        参数:
            user (int): 用户索引
            item (int): 物品索引
            
        返回:
            float: 预测评分
        """
        if self.model is None:
            return 0.0
        
        try:
            # 检查输入有效性
            if not isinstance(user, (int, np.integer)) or not isinstance(item, (int, np.integer)):
                print(f"预测时输入类型错误: user={type(user)}, item={type(item)}")
                return 0.0
                
            if user < 0 or user >= self.n_users or item < 0 or item >= self.n_items:
                print(f"预测时索引超出范围: user={user}, item={item}")
                return 0.0
            
            # 转换为PyTorch张量
            user_tensor = torch.LongTensor([user]).to(self.device)
            item_tensor = torch.LongTensor([item]).to(self.device)
            
            # 预测
            with torch.no_grad():
                prediction = self.model(user_tensor, item_tensor)
            
            # 确保结果是有效的浮点数
            result = float(prediction.item())
            if not np.isfinite(result):
                print(f"预测结果无效: {result}")
                return 0.0
                
            return result
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            return 0.0
    
    def recommend(self, user: int, top_k: int = 10) -> list:
        """
        为用户推荐物品
        
        参数:
            user (int): 用户索引
            top_k (int): 推荐物品数量
            
        返回:
            list: 推荐物品索引列表
        """
        if self.user_item_matrix is None or self.model is None:
            return []
        
        # 获取用户未评分的物品
        user_ratings = self.user_item_matrix[user]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # 如果没有未评分的物品，返回空列表
        if len(unrated_items) == 0:
            return []
        
        # 使用批处理方式预测评分
        try:
            # 创建批次以避免内存溢出
            batch_size = 1024
            n_items = len(unrated_items)
            scores = np.zeros(n_items)
            
            for i in range(0, n_items, batch_size):
                batch_items = unrated_items[i:i+batch_size]
                
                # 创建用户和物品张量
                user_tensor = torch.LongTensor([user] * len(batch_items)).to(self.device)
                item_tensor = torch.LongTensor(batch_items).to(self.device)
                
                # 批量预测
                with torch.no_grad():
                    batch_scores = self.model(user_tensor, item_tensor).cpu().numpy()
                
                # 保存分数
                scores[i:i+len(batch_items)] = batch_scores
            
            # 选择评分最高的top_k个物品
            top_indices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
            return [unrated_items[i] for i in top_indices]
            
        except Exception as e:
            print(f"批量推荐过程中出错: {str(e)}")
            # 回退到逐个预测
            try:
                scores = []
                for item in unrated_items[:min(1000, len(unrated_items))]:
                    score = self.predict(user, item)
                    scores.append(score)
                
                # 选择评分最高的top_k个物品
                top_indices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
                return [unrated_items[i] for i in top_indices]
            except Exception as e:
                print(f"回退推荐过程中出错: {str(e)}")
                return []


class NCFModel(nn.Module):
    """
    NCF模型的PyTorch实现
    """
    def __init__(self, n_users, n_items, n_factors=8, layers=[16, 8], dropout=0.2, alpha=0.5):
        """
        初始化NCF模型
        
        参数:
            n_users (int): 用户数量
            n_items (int): 物品数量
            n_factors (int): 隐向量维度
            layers (list): MLP层的神经元数量列表
            dropout (float): Dropout比例
            alpha (float): GMF和MLP的融合权重
        """
        super(NCFModel, self).__init__()
        
        # GMF部分
        self.user_embedding_gmf = nn.Embedding(n_users, n_factors)
        self.item_embedding_gmf = nn.Embedding(n_items, n_factors)
        
        # MLP部分
        self.user_embedding_mlp = nn.Embedding(n_users, n_factors)
        self.item_embedding_mlp = nn.Embedding(n_items, n_factors)
        
        # MLP层
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * n_factors
        for i, layer_size in enumerate(layers):
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size
        
        # 输出层
        self.output_layer = nn.Linear(n_factors + layers[-1], 1)
        
        # 融合权重
        self.alpha = alpha
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
    
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
            user_indices (torch.Tensor): 用户索引
            item_indices (torch.Tensor): 物品索引
            
        返回:
            torch.Tensor: 预测评分
        """
        # GMF部分
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embedding_gmf * item_embedding_gmf
        
        # MLP部分
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        
        # MLP层
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        
        # 融合GMF和MLP
        concat_output = torch.cat([gmf_output, mlp_input], dim=-1)
        
        # 输出层
        prediction = self.output_layer(concat_output)
        
        return prediction.view(-1)