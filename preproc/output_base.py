# moar garbage. set output path.

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter


def get_parser():
    parser = ArgumentParser(description='Set output path',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sub', '--sub',
                        action='store', help='sub prefix')

    parser.add_argument('--ses', '--ses',
                        action='store', help='ses prefix')

    parser.add_argument('--run', '--run',
                        action='store', help='run prefix')

    parser.add_argument('--task', '--task',
                        action='store', help='task prefix')

    parser.add_argument('--out', '--out',
                        action='store', help='Output directory')

    return parser

def main():
    """Entry point"""
    opts = get_parser().parse_args()
    root = '{}/'.format(opts.out)
    if opts.sub:
        root = '{}{}'.format(root, opts.sub)

    if opts.ses:
        root = '{}_{}'.format(root, opts.ses)

    if opts.task:
        root = '{}_{}'.format(root, opts.task)

    if opts.run:
        root = '{}_{}'.format(root, opts.run)

    print(root)


if __name__ == '__main__':
    main()
