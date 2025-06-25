# 推荐系统框架

这是一个模块化的推荐系统框架，实现了基于用户和基于商品的协同过滤算法，并提供了灵活的架构以便于添加新的推荐模型（如矩阵分解等）。

## 项目结构

- `data_loader.py`: 数据加载和预处理模块
- `models.py`: 推荐模型定义模块
- `metrics.py`: 评估指标模块
- `pipeline.py`: 推荐系统流程模块
- `main.py`: 入口文件，展示如何使用框架
- `Musical_Instruments.csv`: 示例数据集（亚马逊乐器评论数据集）

## 功能特点

1. **模块化设计**：各个组件（数据加载、模型、评估指标、流程）分离，便于扩展和维护
2. **可扩展的模型架构**：通过抽象基类`RecommenderModel`，可以轻松添加新的推荐算法
3. **完整的评估体系**：包含多种评估指标（MAE、RMSE、准确率、召回率、F1等）
4. **简单易用的接口**：提供统一的训练、评估和推荐接口

## 已实现的模型

1. **基于用户的协同过滤（UserBasedCF）**：根据用户之间的相似度进行推荐
2. **基于商品的协同过滤（ItemBasedCF）**：根据商品之间的相似度进行推荐

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python main.py
```

### 使用自定义数据集

修改`main.py`中的数据路径：

```python
data_path = "path/to/your/dataset.csv"
```

确保您的数据集包含以下列：
- `userID`: 用户ID
- `itemID`: 商品ID
- `rating`: 评分

### 添加新模型

1. 在`models.py`中创建新的模型类，继承`RecommenderModel`基类
2. 实现`fit`和`predict`方法
3. 在`main.py`中使用新模型

示例：

```python
class MatrixFactorization(RecommenderModel):
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=0.1, name="MatrixFactorization"):
        super().__init__(name=name)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        # 其他初始化代码
    
    def fit(self, user_item_matrix):
        # 实现矩阵分解训练逻辑
        self.is_fitted = True
        return self
    
    def predict(self, user_idx, item_idx=None, top_n=10):
        # 实现预测逻辑
        pass
```

然后在`main.py`中使用：

```python
# 创建矩阵分解模型
mf = MatrixFactorization(n_factors=50)

# 创建推荐流程
mf_pipeline = RecommenderPipeline(
    data_path=data_path,
    model=mf
)

# 训练和评估
mf_pipeline.train_model()
mf_results = mf_pipeline.evaluate_model()
```

## 扩展建议

1. **添加更多模型**：
   - 矩阵分解（Matrix Factorization）
   - 基于内容的推荐（Content-based Recommendation）
   - 神经网络模型（Neural Collaborative Filtering）

2. **增强数据处理**：
   - 处理冷启动问题
   - 添加特征工程
   - 支持更多数据格式

3. **改进评估系统**：
   - 添加交叉验证
   - 实现更多评估指标
   - 添加超参数调优

## 依赖库

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn