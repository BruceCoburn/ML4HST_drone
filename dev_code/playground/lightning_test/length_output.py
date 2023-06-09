# Create a python script which calculates the output length of an image assuming the user inputs the image length,
# kernel size, stride and padding. The formula is as follows:
# output_length = (input_length - kernel_size + 2 * padding) / stride + 1
#
# The script should be able to be run from the command line as follows:
# python length_output.py --input_length 224 --kernel_size 7 --stride 2
# --padding 1

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculate the output length of an image')
    parser.add_argument(
        '--input_length',
        type=int,
        help='Input length of the image')
    parser.add_argument('--kernel_size', type=int, help='Kernel size')
    parser.add_argument('--stride', type=int, help='Stride')
    parser.add_argument('--padding', type=int, help='Padding')

    args = parser.parse_args()

    output_length = (args.input_length - args.kernel_size +
                     2 * args.padding) / args.stride + 1

    print('Output length: %d' % output_length)
