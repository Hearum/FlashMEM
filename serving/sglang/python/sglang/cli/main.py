# Modified for the anonymous serving artifact.
"""Command-line interface for the CUDA language-model serving distribution."""

import argparse

from sglang.version import __version__


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Launch the language model server.", add_help=False)
    commands.add_parser("version", help="Show the package version.")
    args, server_args = parser.parse_known_args()
    if args.command == "version":
        if server_args:
            parser.error("The version command does not accept arguments.")
        print(f"sglang version: {__version__}")
    else:
        from sglang.launch_server import main as launch

        launch(server_args)
