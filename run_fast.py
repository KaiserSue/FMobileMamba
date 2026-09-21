"""Independent classification training entry point."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--cfg_path', default='configs/mobilemamba/mobilemamba_t2.py')
    parser.add_argument('-m', '--mode', default='train', choices=['train', 'ft', 'val', 'test', 'test_net'])
    parser.add_argument('--sleep', type=int, default=-1)
    parser.add_argument('--memory', type=int, default=-1)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--logger_rank', type=int, default=0)
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parent)
    from fast_cls import run
    run(args)


if __name__ == '__main__':
    main()
