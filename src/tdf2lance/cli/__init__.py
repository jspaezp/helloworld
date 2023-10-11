# SPDX-FileCopyrightText: 2023-present J. Sebastian Paez <jspaezp@gmail.com>
#
# SPDX-License-Identifier: MIT
import click

from tdf2lance.__about__ import __version__
from tdf2lance.conversion import convert_bruker


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="tdf2lance")
@click.argument("bruker_dir_path", type=click.Path(exists=True, file_okay=False))
@click.argument("output_ds", type=click.Path(exists=False), default="output.lance")
def tdf2lance(bruker_dir_path, output_ds):
    convert_bruker(bruker_directory=bruker_dir_path, output_ds=output_ds)
