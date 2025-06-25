import argparse
from pipeline import Pipeline
from data_loader import DataLoader
from models.user_cf import UserCF
from models.item_cf import ItemCF
from models.matrix_factorization import MatrixFactorization
from models.ncf import NCF

MODEL_MAP = {
    'user_cf': UserCF,
    'item_cf': ItemCF,
    'matrix_factorization': MatrixFactorization,
    'ncf': NCF
}

def main():
    parser = argparse.ArgumentParser(description='协同过滤推荐系统')
    parser.add_argument('--data', type=str, default='Musical_Instruments.csv', help='数据集路径')
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='user_cf', help='选择模型')
    parser.add_argument('--topk', type=int, default=10, help='推荐商品数量')
    parser.add_argument('--user', type=str, help='为指定用户推荐商品')
    args = parser.parse_args()

    model_kwargs = {}
    if args.model == 'ncf':
        # 实例化DataLoader来获取num_users和num_items
        data_loader_for_ncf = DataLoader(args.data)
        data_loader_for_ncf.load_data()
        data_loader_for_ncf.preprocess_data()
        user_mapping, item_mapping, _, _ = data_loader_for_ncf.get_mappings()
        model_kwargs = {
            'num_users': len(user_mapping),
            'num_items': len(item_mapping)
        }

    pipeline = Pipeline(args.data, MODEL_MAP[args.model], model_kwargs=model_kwargs)
    pipeline.run()
    if args.user:
        recs = pipeline.recommend_for_user(args.user, args.topk)
        print(f'为用户 {args.user} 推荐的商品ID: {recs}')

if __name__ == '__main__':
    main()