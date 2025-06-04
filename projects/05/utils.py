from pathlib import Path

import numpy as np
from scipy.io import wavfile

from typing import List, Tuple


def simulate_awgn_channel(x: List[float], snr_db: float) -> Tuple[List[float], float]:
    """
    Simulates the passage of a signal through an AWGN channel with random attenuation.

    :param x: Input signal (encoded, ex: Manchester, AMI, etc.)
    :param snr_db: Signal-to-noise ratio (in decibels)
    :return: Received signal y(k) and the attenuation alpha used
    """
    x = np.array(x)

    alpha = np.random.uniform(0.0, 1.0)
    attenuated_signal = alpha * x

    signal_power = np.mean(attenuated_signal ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), len(x))

    y = attenuated_signal + noise

    return y.tolist(), alpha


def get_pcm_8_encoded_wav(file: Path) -> Tuple[bytearray, float]:
    """
    Reads a WAV file and returns the data as a bytearray (8 bits PCM).

    :param file: Path to the WAV file
    :return: Data as a bytearray (8 bits PCM) and the sample rate
    """

    sample_rate, data = wavfile.read(file)

    data_max = np.max(data)
    data_min = np.min(data)

    normalized_data = ((data - data_min) / (data_max - data_min) * 255).astype('uint8')

    return bytearray(normalized_data), sample_rate


def get_ber(original_data: bytearray, received_data: bytearray) -> np.float64:
    """
    Calculate the bit error rate (BER) between two bytearrays (Compares bit by bit).

    :param original_data: Original data as a bytearray
    :param received_data: Received data as a bytearray
    :return: Bit error rate (BER)
    """

    assert len(original_data) == len(received_data), "[get_ber] Original and received data must have the same length"

    total_bits = len(original_data) * 8
    error_bits = 0

    for i, original_byte in enumerate(original_data):
        error_bits += bin(original_byte ^ received_data[i]).count('1')

    return np.float64(error_bits) / np.float64(total_bits) if total_bits > 0 else 0.0


def validate_files(files: List[Path]) -> List[Path]:
    """
    Validates a list of files.
        :param files: List of files to validate
        :return: List of valid files
    """
    valid_files: List[Path] = []
    for file in files:
        if not file.exists():
            print("File {} does not exist.", file)
            continue
        if file.suffix.lower() != ".wav":
            print("File {} is not a WAV file.", file)
            continue
        valid_files.append(file)
    return valid_files
