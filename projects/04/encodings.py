import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def plot_encoded_signal(signal: Tuple[List[float], List[float]], title: str, save_to_file: Optional[str] = None) -> None:
    """
    Plot the encoded signal
    :param signal: Tuple containing the signal values and time values
    :param title: Title of the plot
    :param save_to_file: Optional filename to save the plot
    :return: None
    """
    fig, ax = plt.subplots()
    ax.plot(signal[1], signal[0], drawstyle='steps-post')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.grid(True, alpha=0.5)
    if save_to_file:
        fig.savefig(save_to_file)
    plt.show()


def nrz_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, levels: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
    """
    Non-Return-to-Zero (NRZ) Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :param low_level: Low level voltage for '0'
    :return: [List of encoded signal values, time values]
    """

    if levels is None:
        levels = [1.0, 0.0]
    n_bits = np.log2(len(levels))
    assert n_bits.is_integer(), "Number of levels must be a power of 2."
    assert

    lookup = {format(i, f'0{int(n_bits)}b'): levels[i] for i in range(len(levels))}

    signal = []
    # Iterates over the data gathering n_bits at a time
    for byte in data:
        for i in range(0, 8, int(n_bits)):
            bits = format(byte, '08b')[i:i + int(n_bits)]
            signal.extend([lookup[bits]] * samples_per_symbol)

    sample_time = symbol_time / samples_per_symbol
    return signal, np.arange(0, len(signal) * sample_time, sample_time).tolist()


def rz_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, high_level: float = 1.0, low_level: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    Return-to-Zero (RZ) Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :param low_level: Low level voltage for '0'
    :return: [List of encoded signal values, time values]
    """
    signal = []
    for byte in data:
        for bit in format(byte, '08b'):
            if bit == '1':
                signal.extend([high_level] * (samples_per_symbol // 2))
                signal.extend([0.0] * (samples_per_symbol // 2))
            else:
                signal.extend([low_level] * (samples_per_symbol // 2))
                signal.extend([0.0] * (samples_per_symbol // 2))
    sample_time = symbol_time / samples_per_symbol
    return signal, np.arange(0, len(signal) * sample_time, sample_time).tolist()


def rz_unipolar_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, high_level: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Return-to-Zero (RZ) Unipolar Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :return: [List of encoded signal values, time values]
    """
    return rz_encoding(data, symbol_time, samples_per_symbol, high_level, low_level=0.0)


def nrz_unipolar_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, high_level: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Non-Return-to-Zero (NRZ) Unipolar Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :return: [List of encoded signal values, time values]
    """
    return nrz_encoding(data, symbol_time, samples_per_symbol, high_level, low_level=0.0)


def nrz_polar_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, high_level: float = 1.0, low_level: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """
    Non-Return-to-Zero (NRZ) Polar Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :param low_level: Low level voltage for '0', defaults to negative of high_level
    :return: [List of encoded signal values, time values]
    """
    if low_level is None:
        low_level = -high_level
    return nrz_encoding(data, symbol_time, samples_per_symbol, high_level, low_level)


def rz_polar_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int, high_level: float = 1.0, low_level: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """
    Return-to-Zero (RZ) Polar Encoding
    :param data: Input data as a bytearray
    :param symbol_time: Time duration for each symbol
    :param samples_per_symbol: Number of samples per symbol
    :param high_level: High level voltage for '1'
    :param low_level: Low level voltage for '0', defaults to negative of high_level
    :return: [List of encoded signal values, time values]
    """
    if low_level is None:
        low_level = -high_level
    return rz_encoding(data, symbol_time, samples_per_symbol, high_level, low_level)

data = bytearray([0b11011000])

plot_encoded_signal(
    nrz_encoding(data, 0.1, 10, np.linspace(0.0, 5.0, 8).tolist()),
    "Test",
    "test.png"
)
