import argparse
from applyaudiosr import AudioSuperResolutionWAVProcessor


def process():
    parser = argparse.ArgumentParser(
        description="This script performs the automation required to process a large WAV file with AudioSR"
    )

    # Waveform path argument
    parser.add_argument(
        "--waveform-path",
        type=str,
        required=True,
        help="The path to the WAV file to be processed",
    )
    # Guidance scale argument
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="The guidance scale to be used in AudioSR",
    )
    # Seed argument
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed to be used in AudioSR",
    )
    # DDIM steps argument
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="The number of DDIM steps to be used in AudioSR",
    )
    # Model name argument
    parser.add_argument(
        "--model-name",
        type=str,
        default="basic",
        help="The model name to be used in AudioSR",
    )

    # Parse the arguments
    args = parser.parse_args()

    audio_sr_wav_processor = AudioSuperResolutionWAVProcessor(
        waveform_path=args.waveform_path,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        ddim_steps=args.ddim_steps,
        model_name=args.model_name,
    )
    seed = audio_sr_wav_processor.process()

    print("Processed audio file with seed:", seed)
