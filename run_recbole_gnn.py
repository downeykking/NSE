import sys
sys.path.append(".")
import argparse

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LightGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='yelp', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    parser.add_argument('--gpu_id', type=str, default='0', help='config files')

    parser.add_argument('--embedding_size', type=int, default=64, help='channels of embedding representation. (default: 64)')
    parser.add_argument("--n_layers", type=int, default=2, help='layers of encoder. (default: 2)')
    parser.add_argument('--reg_weight', type=float, default=1e-6,
                        help='regularization weight on embeddings, use in EmbLoss. (default: 1e-6)')

    parser.add_argument('--hidden_size_list', nargs="+", type=int,
                        default=[64, 64, 64], help='a list of hidden representation channels.')
    parser.add_argument('--node_dropout', type=float, default=0.1, help='node dropout rate.')
    parser.add_argument('--message_dropout', type=float, default=0.1, help='edge dropout rate.')
    parser.add_argument('--not_require_pow', default=False, action='store_true',
                        help="don't power the embeddings' norm.(default: False)")
    parser.add_argument('--aug_type', type=str, default="ED",
                        help='augmentation of contrastive learning used in sgl, must in ["ND", "ED", "RW"].')
    parser.add_argument('--drop_ratio', type=float, default=0, help='drop rate in contrastive learning.')
    parser.add_argument('--ssl_tau', type=float, default=0.2, help='temperature in contrastive learning.')
    parser.add_argument('--ssl_weight', type=float, default=0.001, help='the weight of contrastive learning loss.')

    parser.add_argument('--epochs', type=int, default=500, help='train epoch.')
    parser.add_argument('--use_relu', default=False, action='store_true', help='relu or identity.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate.')

    parser.add_argument('--ptb_strategy', type=str, default="none", help='polluted strategy')
    parser.add_argument('--ptb_prop', type=float, default=0.05, help='polluted rate')
    parser.add_argument('--warm_up_step', type=int, default=0, help='warm_up_step.')

    parser.add_argument('--stopping_step', type=int, default=10, help='stopping_step')

    args = parser.parse_args()

    config = {
        "gpu_id": args.gpu_id,
        "epochs": args.epochs,
        "use_relu": args.use_relu,
        "n_layers": args.n_layers,
        "hidden_size_list": args.hidden_size_list,
        "node_dropout": args.node_dropout,
        "message_dropout": args.message_dropout,
        "embedding_size": args.embedding_size,
        "reg_weight": args.reg_weight,
        "require_pow": not args.not_require_pow,
        "aug_type": args.aug_type,
        "drop_ratio": args.drop_ratio,
        "ssl_tau": args.ssl_tau,
        "ssl_weight": args.ssl_weight,
        "learning_rate": args.learning_rate,
        'ptb_strategy': str(args.ptb_strategy),
        'ptb_prop': args.ptb_prop,
        'warm_up_step': args.warm_up_step,
        'stopping_step': args.stopping_step,
    }

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config)
