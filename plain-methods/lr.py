import argparse
from ..tools import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--val_filepath', type=str, default='./data/training1109.xlsx', help='输入验证数据集路径')
    parser.add_argument('--save_filename', type=str, default='results', help='结果输出文件名')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
  args = parse_args_and_config()
  features, labels = read_fea_label(args.val_filepath)