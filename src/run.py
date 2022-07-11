from argparse import Namespace, ArgumentParser
from main import main
import torch
import ast
import os

def arg_update(config):
    # Dims
    config.dims = ast.literal_eval(config.dims)
    # Number per layers
    config.num_per_layers = ast.literal_eval(config.num_per_layers)
    # GPU checking
    if not torch.cuda.is_available() or config.gpuid == -1: config.device = 'cpu'
    else: config.device = f'cuda:{config.gpuid}' 
    # Shuffle setting
    config.shuffle = False if config.shuffle == 'False' else True
    # Save folder
    if not os.path.exists(config.save_path): 
        os.makedirs(config.save_path)
    print(f'Save to: {config.save_path}')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    # Model configuration
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--dims', type=str)
    parser.add_argument('--num_per_layers', type=str)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--inv_kernel', type=int, default=7)

    # Train configuration
    ## Device
    parser.add_argument('--gpuid', type=int, default=0)
    ## Directory
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--train_image_dir', type=str, default='data/train_set/images/')
    parser.add_argument('--test_image_dir', type=str, default='data/val_set/images/')
    ## File name
    parser.add_argument('--train_input', type=str, default='train_subset_20000_reg.csv')
    parser.add_argument('--test_input', type=str, default='test_reg.csv')
    ## Procedure
    parser.add_argument('--shuffle', choices=('True', 'False'), default='False')
    parser.add_argument('--subset', type=int, default=None)
    ## Saving configuration
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_name', type=str)
    ## Speed up argument
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accummulative_iteration', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    ## Epoch
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=50)
    ## Classification or Regression
    parser.add_argument('--mode', choices=('reg', 'class', 'class_reg'), default='reg')
    ## Train & Test
    parser.add_argument('--task', choices=('train', 'test'), default='train')
    ## Transfer file
    parser.add_argument('--transfer_file', type=str, default=None)
    parser.add_argument('--optimizer', choices=('Adam', 'SGD'), default='SGD')

    args = parser.parse_args()
    
    config = Namespace(
        # Model parameters
        in_channel=args.in_channel, # Inital input dimension
        dims=args.dims, # Output dimension
        num_per_layers=args.num_per_layers, # Number of layers per output dimension
        dropout_rate=args.dropout_rate, # Dropout rate
        inv_kernel=args.inv_kernel, # Involution kernel
        batch_size=args.batch_size, # Batch size
        accummulative_iteration=args.accummulative_iteration, # Batch increase

        # Computer configuration
        gpuid=args.gpuid,  # Gpu
        num_workers=args.num_workers, # Workers for faster datalaoder

        # Data configuration
        root=args.root, # Data root directory
        train_image_dir=args.train_image_dir, # Train directory
        test_image_dir=args.test_image_dir, # Test directory
        train_input=args.train_input, # Train input file
        test_input=args.test_input, # Test input file
        shuffle=args.shuffle, # Shuffle the dataset
        subset=args.subset, # Subset for testing

        # Mode and task selection
        mode=args.mode, # class or reg
        task=args.task, # train or test

        # Weights saving configuration
        save_path=args.save_path, # Save path
        save_name=args.save_name, # Save name

        # Train and test configuration
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch, 

        # Transfer learning classification
        transfer_file=args.transfer_file, # Transfer save file
        optimizer=args.optimizer # Switching between Adam and SGD
    )

    # Argument update
    arg_update(config)
    main(config)



