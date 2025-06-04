from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CommandLineArgs:
    min_snr_db: float
    max_snr_db: float
    samples_per_snr: int
    samples_per_symbol: int
    files: List[Path]

    @staticmethod
    def from_args() -> "CommandLineArgs":
        parser = ArgumentParser(description="BER Analysis tool.")

        parser.add_argument(
            "--min_snr_db",
            type=float,
            default=5.0,
            help="Minimum SNR in dB for analysis.",
        )
        parser.add_argument(
            "--max_snr_db",
            type=float,
            default=45.0,
            help="Maximum SNR in dB for analysis.",
        )
        parser.add_argument(
            "-n",
            "--samples_per_snr",
            type=int,
            default=200,
            help="Number of samples per SNR value.",
        )
        parser.add_argument(
            "-s",
            "--samples_per_symbol",
            type=int,
            default=2,
            help="Number of samples per symbol.",
        )
        parser.add_argument(
            "-f",
            "--files",
            nargs="+",
            type=Path,
            default=[],
            help="List of WAV files to analyze.",
        )

        args = parser.parse_args()
        client_args = CommandLineArgs(**vars(args))

        return client_args
