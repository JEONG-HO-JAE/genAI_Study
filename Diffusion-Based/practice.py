from pathlib import Path  
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', type=Path, required=True, help='Path to the checkpoint file')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to use')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('-w', type=float, default=None, help='Class guidance')
    parser.add_argument('-T', type=int, default=None, help='Number of diffusion steps')
    return parser.parse_args()

def main():
    """
    Generate images from a trained model in the checkpoint folder
    """
    args = parse_args()
    print(args)
    run_path = args.run.resolve()  # abspath 대신 resolve()를 사용합니다.
    print(run_path)
    
    
if __name__ == '__main__':
    main()