"""Argument parsing."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        nargs="?",
                        default="../input/qed/positive/train/",
	                help="Folder with training graph jsons.")

    parser.add_argument("--property",
                        type = str,
                        default= 'qed')

    parser.add_argument("--use_mi",
                        action = 'store_true')

    parser.add_argument("--unsupervised",
                        action = 'store_true',
                        help="Folder with training graph jsons.")

    parser.add_argument("--train_percent",
                        type = float,
                        default= 0.85,
                        help="Folder with training graph jsons.")

    parser.add_argument("--validate_percent",
                        type = float,
                        default= 0.05,
                        help="Folder with training graph jsons.")

    parser.add_argument("--subgraph_const",
                        type = float,
                        default= 0.8 ,
                        help="Folder with training graph jsons.")

    parser.add_argument("--first-gcn-dimensions",
                        type=int,
                        default=16,
	                help="Filters (neurons) in 1st convolution. Default is 32.")

    parser.add_argument("--second-gcn-dimensions",
                        type=int,
                        default=16,
	                help="Filters (neurons) in 2nd convolution. Default is 16.")

    parser.add_argument("--first-dense-neurons",
                        type=int,
                        default=16,
	                help="Neurons in SAGE aggregator layer. Default is 16.")

    parser.add_argument("--second-dense-neurons",
                        type=int,
                        default=2,
	                help="SAGE attention neurons. Default is 8.")


    parser.add_argument("--epochs",
                        type=int,
                        default=2,
	                help="Number of epochs. Default is 10.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-5,
	                help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--gamma",
                        type=float,
                        default=10**-5,
	                help="Attention regularization coefficient. Default is 10^-5.")

    parser.add_argument("--save",
                        type=str,
                        default='../test_results/qed_positive/',
                        help="save results .")

    parser.add_argument("--batch_size",
                        type=int,
                        default= 128,
                        help="batch_size")

    parser.add_argument("--cls_hidden_dimensions",
                        type=int,
                        default= 4,
                        help="classifier hidden dims")

    parser.add_argument("--dis_hidden_dimensions",
                        type=int,
                        default= 4,
                        help="classifier hidden dims")

    parser.add_argument("--mi_weight",
                        type=float,
                        default= 0.0001,
                        help="classifier hidden dims")

    parser.add_argument("--con_weight",
                        type=float,
                        default= 5,
                        help="classifier hidden dims")

    parser.add_argument("--inner_loop",
                        type=int,
                        default= 50,
                        help="classifier hidden dims")

    parser.add_argument("--noise_scale",
                        type=float,
                        default= 0.1,
                        help="classifier hidden dims")

    parser.add_argument("--warm_up",
                        type=int,
                        default= 1,
                        help="classifier hidden dims")

    parser.add_argument("--gnn",
                        type=str,
                        default= 'GCN',
                        help="classifier hidden dims")

    parser.add_argument("--training_dir",
                        type=str,
                        default= 'training/',
                        help="classifier hidden dims")

    parser.add_argument("--testing_dir",
                        type=str,
                        default= 'testing/',
                        help="classifier hidden dims")



    return parser.parse_args()
