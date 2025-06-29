import argparse
from pipeline import Pipeline
from data_loader import DataLoader
from models.user_cf import UserCF
from models.item_cf import ItemCF
from models.matrix_factorization import MatrixFactorization

MODEL_MAP = {
    'user_cf': UserCF,
    'item_cf': ItemCF,
    'matrix_factorization': MatrixFactorization
}

def main():
    parser = argparse.ArgumentParser(description='协同过滤推荐系统')
    parser.add_argument('--data', type=str, default='Musical_Instruments.csv', help='数据集路径')
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='user_cf', help='选择模型')
    parser.add_argument('--topk', type=int, default=10, help='推荐商品数量')
    parser.add_argument('--user', type=str, help='为指定用户推荐商品')
    parser.add_argument('--detail', action='store_true', help='显示详细的推荐信息，包括用户实际评分最高的商品和预测喜欢的商品')
    args = parser.parse_args()

    model_kwargs = {}
    if args.model == 'matrix_factorization':
        # 为矩阵分解模型设置参数
        model_kwargs = {
            'n_factors': 100,  # 隐因子数量
            'n_epochs': 20,   # 训练轮数
            'lr': 0.005,      # 学习率
            'reg': 0.02       # 正则化系数
        }

    pipeline = Pipeline(args.data, MODEL_MAP[args.model], model_kwargs=model_kwargs)
    pipeline.run()
    if args.user:
        if args.detail:
            # 显示详细的推荐信息
            pipeline.print_user_recommendations(args.user, args.topk)
        else:
            # 只显示推荐的商品ID列表
            recs = pipeline.recommend_for_user(args.user, args.topk)
            print(f'为用户 {args.user} 推荐的商品ID: {recs}')

if __name__ == '__main__':
    main()