import os, sys
from parsers import main_parser
from commands.split import Split
from commands.train import Train
from commands.btirads2 import BTIRADS2
from commands.evaluate import Evaluate
from commands.shap_analysis import Shap_analysis


def main():
    parser, subparsers = main_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return
    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
            subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
            return

    args = parser.parse_args()

    # limit the cpu core usage
    n_cpus = args.cpu_range[-1] - args.cpu_range[0]
    pid = os.getpid()

    cpu_arg = "".join(
        [str(ci) + "," for ci in range(args.cpu_range[0], args.cpu_range[-1])]
    )[:-1]
    cmd = "taskset -cp %s %i >/dev/null 2>&1" % (cpu_arg, pid)

    os.system(cmd)

    args.n_jobs = n_cpus

    if args.command == "split":
        command = Split(args)
    elif args.command == "train":
        command = Train(args)
    elif args.command == "btirads2":
        command = BTIRADS2(args)
    elif args.command == "shap_analysis":
        command = Shap_analysis(args)
    elif args.command == "evaluate":
        command = Evaluate(args)
    else:
        raise "Command not supported"

    command.exec()


if __name__ == "__main__":
    main()
