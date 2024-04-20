import unittest
from unittest.mock import patch
from applyaudiosr import AudioSuperResolutionWAVProcessor
from applyaudiosr.constants import (
    PROCESSED_FILE_SUFFIX,
)

class TestAudioSuperResolutionWAVProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioSuperResolutionWAVProcessor(waveform_path='/path/to/waveform.wav')

    def test_process(self):
        with patch.object(self.processor, '_clear_base_output_dir') as mock_clear_dir, \
             patch.object(self.processor, 'generate_audio_chunk_batch_list_file_from_waveform') as mock_generate_file, \
             patch('subprocess.run') as mock_subprocess_run, \
             patch.object(self.processor, 'combine_waveforms_in_dir') as mock_combine_waveforms:

            # Mock the return values of the mocked methods
            mock_generate_file.return_value = '/path/to/audio_chunks_batch.txt'

            # Call the method under test
            self.processor.process()

            # Assert that the methods were called with the expected arguments
            mock_clear_dir.assert_called_once()
            mock_generate_file.assert_called_once()
            mock_subprocess_run.assert_called_once_with([
                "audiosr",
                "-il",
                "/path/to/audio_chunks_batch.txt",
                "-s",
                self.processor.audiosr_outputs_dir,
                "--suffix",
                PROCESSED_FILE_SUFFIX,
            ])
            mock_combine_waveforms.assert_called_once()