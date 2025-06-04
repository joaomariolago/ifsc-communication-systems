from typing import List, Tuple
import numpy as np
from numba import njit


@njit
def _manchester_encode_bits(bits: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    half = samples_per_symbol // 2
    signal = np.empty(len(bits) * samples_per_symbol, dtype=np.float32)

    for i in range(len(bits)):
        base = i * samples_per_symbol
        if bits[i] == 0:
            signal[base:base + half] = 1.0
            signal[base + half:base + samples_per_symbol] = -1.0
        else:
            signal[base:base + half] = -1.0
            signal[base + half:base + samples_per_symbol] = 1.0
    return signal


def manchester_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manchester Encoding as per IEEE 802.3
    """
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    signal = _manchester_encode_bits(bits, samples_per_symbol)
    sample_time = symbol_time / samples_per_symbol
    time = np.arange(len(signal)) * sample_time
    return signal, time


@njit
def _ami_encode(bits: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    signal = np.empty(len(bits) * samples_per_symbol, dtype=np.float32)
    polarity = 1.0
    for i in range(len(bits)):
        base = i * samples_per_symbol
        if bits[i] == 1:
            for j in range(samples_per_symbol):
                signal[base + j] = polarity
            polarity *= -1
        else:
            for j in range(samples_per_symbol):
                signal[base + j] = 0.0
    return signal


def ami_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    signal = _ami_encode(bits, samples_per_symbol)
    sample_time = symbol_time / samples_per_symbol
    time = np.arange(len(signal), dtype=np.float32) * sample_time
    return signal, time


@njit
def _rz_polar_encode(bits: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    signal = np.empty(len(bits) * samples_per_symbol, dtype=np.float32)
    half = samples_per_symbol // 2
    polarity = 1.0

    for i in range(len(bits)):
        base = i * samples_per_symbol
        if bits[i] == 1:
            for j in range(half):
                signal[base + j] = polarity
            for j in range(half):
                signal[base + half + j] = 0.0
            polarity *= -1
        else:
            for j in range(samples_per_symbol):
                signal[base + j] = 0.0

    return signal


def rz_polar_encoding(data: bytearray, symbol_time: float, samples_per_symbol: int) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    signal = _rz_polar_encode(bits, samples_per_symbol)
    sample_time = symbol_time / samples_per_symbol
    time = np.arange(len(signal), dtype=np.float32) * sample_time
    return signal, time


@njit
def __manchester_decode_signal(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    half = samples_per_symbol // 2
    num_symbols = len(signal) // samples_per_symbol
    bits = np.empty(num_symbols, dtype=np.uint8)

    for i in range(num_symbols):
        start = i * samples_per_symbol
        mid = start + half
        end = start + samples_per_symbol

        avg1 = 0.0
        for j in range(start, mid):
            avg1 += signal[j]
        avg1 /= half

        avg2 = 0.0
        for j in range(mid, end):
            avg2 += signal[j]
        avg2 /= half

        bits[i] = 1 if avg1 < avg2 else 0

    return bits


def manchester_decoding(signal: List[float], time: List[float], samples_per_symbol: int) -> bytearray:
    signal_np = np.asarray(signal, dtype=np.float32)
    bits = __manchester_decode_signal(signal_np, samples_per_symbol)

    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.pad(bits, (0, 8 - remainder), constant_values=0)

    bytes_out = np.packbits(bits)
    return bytearray(bytes_out)


@njit
def _ami_decode(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    num_symbols = len(signal) // samples_per_symbol
    bits = np.empty(num_symbols, dtype=np.uint8)

    threshold = np.mean(np.abs(signal))

    for i in range(num_symbols):
        base = i * samples_per_symbol
        avg = 0.0
        for j in range(samples_per_symbol):
            avg += signal[base + j]
        avg /= samples_per_symbol

        if abs(avg) < threshold:
            bits[i] = 0
        else:
            bits[i] = 1

    return bits


def ami_decoding(signal: List[float], time: List[float], samples_per_symbol: int) -> bytearray:
    signal_np = np.asarray(signal, dtype=np.float32)
    bits = _ami_decode(signal_np, samples_per_symbol)

    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.pad(bits, (0, 8 - remainder), constant_values=0)

    byte_values = np.packbits(bits)
    return bytearray(byte_values)


@njit
def _rz_polar_decode(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    num_symbols = len(signal) // samples_per_symbol
    bits = np.empty(num_symbols, dtype=np.uint8)
    half = samples_per_symbol // 2

    threshold = np.mean(np.abs(signal))

    for i in range(num_symbols):
        base = i * samples_per_symbol
        avg = 0.0
        for j in range(half):
            avg += signal[base + j]
        avg /= half

        if abs(avg) < threshold:
            bits[i] = 0
        else:
            bits[i] = 1

    return bits


def rz_polar_decoding(signal: List[float], time: List[float], samples_per_symbol: int) -> bytearray:
    signal_np = np.asarray(signal, dtype=np.float32)
    bits = _rz_polar_decode(signal_np, samples_per_symbol)

    remainder = len(bits) % 8
    if remainder != 0:
        bits = np.pad(bits, (0, 8 - remainder), constant_values=0)

    byte_values = np.packbits(bits)
    return bytearray(byte_values)
