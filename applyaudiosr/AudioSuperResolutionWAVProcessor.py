import glob
import logging
import shutil
import subprocess
import os
import glob
import os
import pydub
import random

from applyaudiosr.constants import (
    AUDIO_CHUNKS_BATCH_FILENAME,
    AUDIO_CHUNKS_DIR_NAME,
    AUDIOSR_OUTPUT_DIR_NAME,
    PROCESSED_FILE_SUFFIX,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AudioSuperResolutionWAVProcessor:
    # The AudioSuperResolutionWAVProcessor class is responsible for processing a large WAV file with AudioSR
    # by splitting it into smaller chunks, running AudioSR on each chunk, and combining the resulting waveforms.

    def __init__(
        self,
        waveform_path,
        model_name="basic",
        guidance_scale=0.0,
        seed=None,
        ddim_steps=50,
    ):
        self.waveform_path = waveform_path
        self.filename = os.path.splitext(os.path.basename(waveform_path))[0].split(".")[
            0
        ]
        self.base_output_dir = os.path.join(os.getcwd(), self.filename)
        self.audiosr_output_dir = os.path.join(
            self.base_output_dir, AUDIOSR_OUTPUT_DIR_NAME
        )

        self.model_name = model_name
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.seed = seed if seed else random.randint(1, 9999999)

    def _clear_base_output_dir(self):
        # The 'audiosr' tool exports files to a directory structure like 'output/2024_01_01_12_12_12/<output>.wav'.
        # To simplify the process of locating the output files, we clear the 'output' directory before each run.
        # This ensures that there's only one directory (corresponding to the latest run) in 'output' at any time,
        # eliminating the need for complex folder name parsing when piecing the processed waveforms back together.
        if os.path.exists(self.base_output_dir):
            shutil.rmtree(self.base_output_dir)

        os.mkdir(self.base_output_dir)

    def combine_waveforms_in_dir(self):
        """
        Combines multiple waveform files in a directory into a single audio file.

        This method reads all the waveform files in a directory, sorts them based on their names,
        and concatenates them into a single audio file. It trims the extra 35 milliseconds from
        the end of each chunk before concatenating.
        """
        processed_output_dir = os.path.join(
            self.audiosr_output_dir, next(os.walk(self.audiosr_output_dir))[1][0]
        )

        # Get a list of all the waveform files in the directory
        waveform_files = glob.glob(os.path.join(processed_output_dir, "*.wav"))

        # Sort the files by their names (assuming the names reflect the order of the chunks)
        waveform_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        waveform_files.sort(
            key=lambda x: int(
                "".join(
                    filter(
                        str.isdigit, os.path.basename(x).split("_")[-1].split(".")[0]
                    )
                )
            )
        )

        # Concatenate the audio files
        combined = pydub.AudioSegment.empty()
        for file in waveform_files:
            chunk = pydub.AudioSegment.from_wav(file)

            # Trims the extra 35 milliseconds from the end of each chunk that `audiosr` adds
            chunk = chunk[:-35]

            combined += chunk

        # Export the combined audio
        combined.export(
            os.path.join(
                self.base_output_dir, f"{self.filename}{PROCESSED_FILE_SUFFIX}.wav"
            ),
            format="wav",
        )

    def generate_audio_chunk_batch_list_file_from_waveform(self):
        """
        Generates audio chunks from a waveform and saves the list of paths to a text file.

        Returns:
            str: The file path of the generated audio chunks batch file.
        """

        # `audio_chunk_dir` is a string representing the absolute path to a directory named "audio_chunks" in the current working directory.
        # This directory is intended to store audio chunks that are generated from a larger audio file.
        audio_chunk_dir = os.path.join(self.base_output_dir, AUDIO_CHUNKS_DIR_NAME)
        os.makedirs(audio_chunk_dir, exist_ok=True)

        # Open the WAV file
        waveform = pydub.AudioSegment.from_wav(self.waveform_path)

        # Split the sound into 5-second chunks (which is a limitation of https://github.com/haoheliu/versatile_audio_super_resolution)
        waveform_chunks = waveform[::5000]

        # Save the list of paths to a text file
        audio_chunks_batch_file_path = os.path.join(
            audio_chunk_dir, AUDIO_CHUNKS_BATCH_FILENAME
        )
        with open(audio_chunks_batch_file_path, "w") as file:
            for i, chunk in enumerate(waveform_chunks):
                chunk_filename = f"{i+1}_{self.filename}_chunk.wav"
                logger.info(f"Created {chunk_filename}")
                chunk_file_path = os.path.join(audio_chunk_dir, chunk_filename)
                chunk.export(chunk_file_path, format="wav")
                file.write(f"{chunk_file_path}\n")

        return audio_chunks_batch_file_path

    def process(self) -> int:
        """
        Process the audio file by applying audio super resolution.

        This method clears the base output directory, generates a batch list file from the waveform,
        runs the audiosr command on the audio chunks, and combines the resulting waveforms.
        """

        # Clear the base output directory
        self._clear_base_output_dir()

        # Generate audio chunks from the waveform and save the list of paths to a text file
        audio_chunks_batch_file_path = (
            self.generate_audio_chunk_batch_list_file_from_waveform()
        )

        # Run the audiosr command on the truncated audio parts and save the outputs to the output directory
        subprocess.run(
            [
                "audiosr",
                "-il",
                audio_chunks_batch_file_path,
                "-s",
                self.audiosr_output_dir,
                "--suffix",
                PROCESSED_FILE_SUFFIX,
                "--model",
                self.model_name,
                "-gs",
                str(self.guidance_scale),
                "--seed",
                str(self.seed),
                "--ddim_steps",
                str(self.ddim_steps),
            ]
        )

        # Combine the waveforms into a single audio file in the base output directory
        self.combine_waveforms_in_dir()

        return self.seed
