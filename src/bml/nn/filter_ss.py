import argparse
import glob
import os
import shutil


def get_ss_it(output_txt):
    with open(output_txt, 'r') as f:
        for line in f:
            if line.startswith("step ="):
                ss_it = int(line[7:line.find(',')])
                return ss_it
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ferrox_base_dir", help="this should look something like 'it00000'")
    parser.add_argument("-l", "--ss_lag", help="the number of iterations before steady state to keep",
                        default=1, type=int)

    args = parser.parse_args()

    ss_it = get_ss_it(os.path.join(args.ferrox_base_dir, "output.txt"))
    plt_dirs = sorted(glob.glob(os.path.join(args.ferrox_base_dir, "plt*")))
    if ss_it > -1:
        ss_it = plt_dirs.index(os.path.join(args.ferrox_base_dir, f"plt{ss_it:08g}"))
        bad_plt_dirs = plt_dirs[:max(0, ss_it - args.ss_lag)]
    else:
        bad_plt_dirs = plt_dirs

    hide_dir = os.path.join(args.ferrox_base_dir, "pre_ss")
    os.makedirs(hide_dir, exist_ok=True)
    for plt_dir in bad_plt_dirs:
        shutil.move(plt_dir, hide_dir)


if __name__ == '__main__':
    main()
