import argparse
from pydub import AudioSegment
import os


def get_wav_metadata(waveform_path):
    """
    This function reads the metadata of a WAV file.

    Args:
        waveform_path (str): The path to the WAV file.

    Returns:
        tuple: A tuple containing the sample width, sample rate, number of channels, number of frames, and duration of the WAV file.
    """
    wav = AudioSegment.from_file(file=waveform_path, format="wav")
    sample_width = wav.sample_width
    sample_rate = wav.frame_rate
    num_channels = wav.channels
    num_frames = wav.frame_count()
    duration = num_frames / sample_rate

    return sample_width, sample_rate, num_channels, num_frames, duration


def print_wav_metadata(waveform_path):
    """
    This function prints the metadata of a WAV file
    """
    sample_width, sample_rate, num_channels, num_frames, duration = get_wav_metadata(
        waveform_path=waveform_path
    )

    filename = os.path.basename(waveform_path)
    print("WAV Metadata")
    print(f"Filename: {filename}")
    print(f"Sample Width: {sample_width}")
    print(f"Sample Rate: {sample_rate}")
    print(f"Number of Channels: {num_channels}")
    print(f"Number of Frames: {num_frames}")
    print(f"Duration: {duration} seconds")
