from pyrecode.recode_server import ReCoDeServer
from pyrecode.params import InitParams
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ReCoDe Server')
    parser.add_argument('--source', dest='source', action='store', default='',
                        help='if mode is batch then file to be processed, otherwise path to folder containing data '
                             '(typically inside RAM disk for on-the-fly)')
    parser.add_argument('--calibration_file', dest='calibration_file', action='store', default='',
                        help='path to calibration file')
    parser.add_argument('--out_dir', dest='out_dir', action='store', default='', help='output directory')
    parser.add_argument('--params_file', dest='params_file', action='store', default='', help='path to params file')
    parser.add_argument('--mode', dest='mode', action='store', default='batch', help='batch or stream')
    parser.add_argument('--validation_frame_gap', dest='validation_frame_gap', action='store', type=int, default=-1,
                        help='validation frame gap')
    parser.add_argument('--log_file', dest='log_file', action='store', default='', help='path to log file')
    parser.add_argument('--run_name', dest='run_name', action='store', default='run_1', help='run name')
    parser.add_argument('--verbosity', dest='verbosity', action='store', type=int, default=0, help='verbosity level')
    parser.add_argument('--use_c', dest='use_c', action='store_true', help='')
    parser.add_argument('--max_count', dest='max_count', action='store', type=int, default=1,
                        help='the number of chunks to process')
    parser.add_argument('--chunk_time_in_sec', dest='chunk_time_in_sec', action='store', type=int, default=1,
                        help='seconds of data contained in each chunk')

    args = parser.parse_args()

    init_params = InitParams(args.mode, args.out_dir, image_filename=args.source, directory_path=args.source,
                             calibration_filename=args.calibration_file, params_filename=args.params_file,
                             validation_frame_gap=args.validation_frame_gap, log_filename=args.log_file,
                             run_name=args.run_name, verbosity=args.verbosity, use_c=False, max_count=args.max_count,
                             chunk_time_in_sec=args.chunk_time_in_sec)

    server = ReCoDeServer(args.mode)
    server.run(init_params)
