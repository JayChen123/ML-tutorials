import argparse

arg_parser = argparse.ArgumentParser(description='run model')

arg_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size的值')
arg_parser.add_argument('-max_len', type=int, help='序列长度', default=500)
arg_parser.add_argument('-epoch', '--epoch', default=10, type=int, help='训练的轮数')

args = arg_parser.parse_args()
print(args.batch_size)
print(args.max_len)
print(args.epoch)
