from argparse import ArgumentParser
from main import main

if __name__ == "__main__":
    parser = ArgumentParser()
    # Model configuration
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--dims', type=str)
    parser.add_argument('--drp_rate', type=float, default=0.5)
    # Files configuration
    parser.add_argument('--file_mode', type=str)
    parser.add_argument('--train_input', type=str)
    parser.add_argument('--test_input', type=str)
    # Test or train mode
    parser.add_argument('--mode', type=str)

    args = parser.parse_args()
    main()



