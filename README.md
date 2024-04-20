### applyaudiosr

A simple CLI script that automates the processes necessary to apply audio super resolution to a full-length WAV file

---

Steps to Run:
```
# Create a new conda environment
conda create -n applyaudiosr python=3.9
conda activate applyaudiosr

# Install poetry
pip3 install poetry

# Install packages
poetry install

# Run
poetry run applyaudiosr --waveform-path lofi_sample.wav --guidance-scale 0.5 --seed 12345 --ddim-steps 25 --model-name basic
```

After the process is complete, the outputs will be stored in a folder with the same name as the audio file. Please note that these outputs will be located in the directory where the command is executed.

---

Test:
```
poetry run python -m unittest discover tests
```

Tested on
- macOS (M1)

---

AudioSR processing was made possible by https://github.com/haoheliu/versatile_audio_super_resolution :bow: