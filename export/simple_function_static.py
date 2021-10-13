
import os
import csv
import sys
import argparse

import stable_nalu
from stable_nalu.reader.tensorboard_metric_reader import _everything_default_matcher

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--tensorboard-dir',
                    action='store',
                    type=str,
                    help='Specify the directory for which the data is stored')
parser.add_argument('--csv-out',
                    action='store',
                    type=str,
                    help='Specify the file for which the csv data is stored at')
parser.add_argument('--export-gate',
                    action='store_true',
                    default=False,
                    help='Export the NALU gate value to the csv file')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Export weights from text summary. For SLTR only')
parser.add_argument('--weights-only',
                    action='store_true',
                    default=False,
                    help='Only parses the text_summary metrics (from the TEXT tb heading)')

args = parser.parse_args()

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    allowed_processes = int(os.environ['LSB_DJOB_NUMPROC'])
else:
    allowed_processes = None

def matcher(tag):
    return (
        tag in ['metric/valid/interpolation', 'metric/test/extrapolation', 'W/text_summary'] or
        tag.endswith('nalu/gate/mean') and args.export_gate if args.verbose else 
        tag in ['metric/valid/interpolation', 'metric/test/extrapolation'] or
        tag.endswith('nalu/gate/mean') and args.export_gate
    )

reader = stable_nalu.reader.TensorboardMetricReader(
    args.tensorboard_dir,
    # TODO: include everything if you want to parse weights saved as SCALARS
    metric_matcher=_everything_default_matcher if args.weights_only else matcher,
    step_start=0,
    processes=allowed_processes,
    recursive_weight=args.verbose,
    weights_only=args.weights_only,
)

def main():
    with open(args.csv_out, 'w') as csv_fp:
        for index, df in enumerate(reader):
            df.to_csv(csv_fp, header=(index == 0), index=False)
            csv_fp.flush()


if __name__ == '__main__':
    # Windows OS requires code to be encapsulated int main() otherwise multiprocessing will not work
    main()
