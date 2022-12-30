import argparse

# training configurations
parser = argparse.ArgumentParser(description="configs for target-oriented opinion words extraction")

parser.add_argument('--dataset', type=str, default='14res', help='dataset: laptop14, rest14, rest15 or rest16')
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--n_epoch", type=int, default=100, help="number of training epochs")
parser.add_argument("--early_stop", type=int, default=20, help="number of early stop epochs")
parser.add_argument("--dropout_rate", type=float, default=0.8, help="dropout rate for the embedding layer")
parser.add_argument("--alpha", type=float, default=1.0, help="rate for the negative loss")
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--load_dir', type=str, default='./BiLSTM_case_study/best_model.pt', help='Root dir for loading models.')
parser.add_argument('--print_step', type=int, default=10, help='Print log every k steps in training process.')

# network configurations
# 14lap: lr: 0.00003, pos 100, bilstm_hidden 150
parser.add_argument("--dim_bilstm_hidden", type=int, default=100, help="hidden dimension for the bilstm")
parser.add_argument("--dim_w", type=int, default=300, help="word embedding dimension")
parser.add_argument("--dim_POS", type=int, default=30, help="POS embedding dimension")
parser.add_argument("--dim_position", type=int, default=30, help="Position embedding dimension")
parser.add_argument("--dim_deprel", type=int, default=30, help="Dependency relation embedding dimension")
parser.add_argument("--gcn_layers", type=int, default=0, help="number of GCN layers")

args = parser.parse_args()
