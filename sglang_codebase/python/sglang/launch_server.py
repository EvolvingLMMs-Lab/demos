import argparse

from sglang.srt.server import ServerArgs, launch_server
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args, None)
