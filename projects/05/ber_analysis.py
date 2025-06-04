import threading
import numpy as np
from typing import Callable, List, Tuple, Dict
from pathlib import Path
from args import CommandLineArgs
from utils import get_pcm_8_encoded_wav, simulate_awgn_channel, get_ber
from digital_encodings import manchester_encoding, ami_encoding, rz_polar_encoding, manchester_decoding, ami_decoding, rz_polar_decoding

class BERAnalysis(threading.Thread):
    def __init__(
        self,
        encoding_function: Callable[[bytearray, float, int], Tuple[List[float], List[float]]],
        decoding_function: Callable[[List[float], List[float]], bytearray],
        snr_db: float,
        samples_per_symbol: int,
        samples_per_snr: int,
        data: bytearray,
        symbol_time: float,
    ) -> None:
        super().__init__()
        self.encoding_function = encoding_function
        self.decoding_function = decoding_function
        self.snr_db = snr_db
        self.samples_per_symbol = samples_per_symbol
        self.samples_per_snr = samples_per_snr
        self.ber_values: List[float] = []
        self.result: Dict[str, float] = {}
        self.data = data
        self.symbol_time = symbol_time

    def run(self) -> None:
        for _ in range(self.samples_per_snr):
            encoded_signal, time = self.encoding_function(self.data, self.symbol_time, self.samples_per_symbol)
            received_signal, _ = simulate_awgn_channel(encoded_signal, self.snr_db)
            decoded_signal = self.decoding_function(received_signal, time, self.samples_per_symbol)
            self.ber_values.append(get_ber(self.data, decoded_signal))
            print(f"SNR: {self.snr_db} dB, BER: {np.mean(self.ber_values)}")

    def collect_results(self) -> Dict[str, float]:
        return {
            "encoding": self.encoding_function.__name__,
            "snr_db": self.snr_db,
            "ber_average": np.mean(self.ber_values),
        }

def main() -> None:
    args = CommandLineArgs.from_args()

    threads = []
    for file in args.files:
        data, sample_rate = get_pcm_8_encoded_wav(file)
        symbol_time = 1.0 / sample_rate

        for snr_db in np.arange(args.min_snr_db, args.max_snr_db, 1.0):
            for encoding_pair in [
                (manchester_encoding, manchester_decoding),
                (ami_encoding, ami_decoding),
                (rz_polar_encoding, rz_polar_decoding),
            ]:
                encoding_function, decoding_function = encoding_pair
                ber_analysis = BERAnalysis(
                    encoding_function=encoding_function,
                    decoding_function=decoding_function,
                    snr_db=snr_db,
                    samples_per_symbol=args.samples_per_symbol,
                    samples_per_snr=args.samples_per_snr,
                    data=data,
                    symbol_time=symbol_time,
                )
                threads.append(ber_analysis)

    for i in range(0, len(threads), 10):
        threads_batch = threads[i:i+10]
        for t in threads_batch:
            t.start()
        for t in threads_batch:
            t.join()
            results = t.collect_results()
            with open(f"results/{results['encoding']}.csv", "a") as f:
                f.write(f"{results['snr_db']}, {results['ber_average']}\n")


if __name__ == "__main__":
    main()
