7.1 更新：加入NCF（神经协同过滤）算法，支持MPS加速 ==Pipeline需要优化，考虑删除ndcg（负样本太少）

6.30 更新：加入MF（SVD）算法，加入协同过滤演示图片

第一周
- [ ] 构建baseline模型评测pipeline
- [ ] 实现了基于用户的协同过滤算法
- [ ] 实现了基于商品的协同过滤算法
- [ ] 实现了矩阵分解推荐算法
- [ ] 制定后续工作计划
- [ ] 深度学习环境配置（PyTorch）

|        | MAE    | RMSE   | NDCG@10 |     |
| ------ | ------ | ------ | ------- | --- |
| UserCF | 2.8592 | 2.0313 |         |     |
| ItemCF | 2.8595 | 2.0046 |         |     |
| MF     | 0.7001 | 0.9239 |         |     |
| NCF    | 0.9562 | 1.1714 |         |     |
|        |        |        |         |     |


第二周
- [x] 复现NCF相关算法
- [x] 补充NDCG@10作为评价指标
- [ ] 调研LLM4Rec相关方法，开始尝试具体LLM4Rec代码实现



使用以下命令运行NCF模型：

```bash
python main.py --model ncf --data Musical_Instruments.csv
```

为特定用户推荐商品并显示详细信息：

```bash
python main.py --model ncf --user A2IIIDRK3PRRZY --detail
```

NCF模型参数说明：
- n_factors: 隐向量维度
- layers: MLP层结构
- lr: 学习率
- n_epochs: 训练轮数
- batch_size: 批次大小
- dropout: Dropout比例
- alpha: GMF和MLP的融合权重

第三周 
- [ ] 完成轻量化LLM增强NCF结构实现，优先场景推理，完成后尝试实现特征融合
- [ ] 开始撰写实验节

第四周 
- [ ] 补充并完善冷启动实验设计与结果
- [ ] 绘制实验结果图表，补充消融实验


第五周 
- [ ] 论文定稿以及最后修改
