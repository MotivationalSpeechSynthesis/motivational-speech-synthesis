import sys
import os
import time
import logging
import string
import random
from pathlib import Path
from typing import Optional

import torch
import librosa

from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

submodule_root = Path(__file__).resolve().parent / "emoknob/src/metavoice-src-main"
sys.path.insert(0, str(submodule_root))

# Imports from fam.llm modules 
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.decoders import EncodecDecoder
from fam.llm.fast_inference_utils import build_model, main
from fam.llm.inference import (
    InferenceConfig,
    Model,
    TiltedEncodec,
    TrainedBPETokeniser,
    get_enhancer,
)
from fam.llm.utils import (
    get_default_dtype,
    normalize_text,
)

class MotivationalTTSConfig(BaseModel):
    seed: Optional[int] = Field(
        None,
        description="Optional seed for inference. If not provided, a random seed will be generated."
    )
    intermediate_dir: str = Field(
        "motivational-speech",
        description="Directory for intermediate outputs."
    )
    device: str = Field(
        "cuda:0",
        description="Computation device."
    )
    dtype: str = Field(
        "float16",
        description="Precision to use ('float16' or 'bfloat16')."
    )
    debug: bool = Field(
        False,
        description="Enable debug logging."
    )
    average_speaker_emb_dir: str = Field(
        "average-speaker-embeddings/average-speaker-embeddings_400",
        description="Directory containing average speaker embeddings."
    )

    class Config:
        arbitrary_types_allowed = True

class MotivationalTTSModel:
    def __init__(self, config: MotivationalTTSConfig = None):
        if config is None:
            config = MotivationalTTSConfig()
        self.config = config

        # Setup logging for debugging if enabled
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG if self.config.debug else logging.INFO)

        # Use provided seed or generate a random one.
        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**32 - 1)
            self.logger.debug(f"No seed provided. Generated random seed: {self.config.seed}")
        else:
            self.logger.debug(f"Using provided seed: {self.config.seed}")

        # Create the intermediate directory for any temporary files if needed.
        os.makedirs(self.config.intermediate_dir, exist_ok=True)

        # Get the default dtype from the library (usually "float16" or "bfloat16")
        self.dtype = get_default_dtype()
        self.device = self.config.device

        self.logger.debug("Downloading model snapshot...")
        self.model_dir = snapshot_download(repo_id="metavoiceio/metavoice-1B-v0.1")

        self.logger.debug("Initializing first stage adapter...")
        self.first_stage_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=1024)

        second_stage_ckpt_path = f"{self.model_dir}/second_stage.pt"
        self.logger.debug("Setting up second stage configuration...")
        self.config_second_stage = InferenceConfig(
            ckpt_path=second_stage_ckpt_path,
            num_samples=1,
            seed=self.config.seed,
            device=self.device,
            dtype=self.dtype,
            compile=False,
            init_from="resume",
            output_dir=self.config.intermediate_dir,
        )

        self.logger.debug("Initializing second stage adapter...")
        self.data_adapter_second_stage = TiltedEncodec(end_of_audio_token=1024)

        self.logger.debug("Loading second stage model...")
        self.llm_second_stage = Model(
            self.config_second_stage,
            TrainedBPETokeniser,
            EncodecDecoder,
            data_adapter_fn=self.data_adapter_second_stage.decode,
        )

        self.logger.debug("Getting speech enhancer...")
        self.enhancer = get_enhancer("df")

        precision_mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        self.precision = precision_mapping[self.dtype]

        self.logger.debug("Building first stage model...")
        self.model, self.tokenizer, self.smodel, self.model_size = build_model(
            precision=self.precision,
            checkpoint_path=Path(f"{self.model_dir}/first_stage.pt"),
            spk_emb_ckpt_path=Path(f"{self.model_dir}/speaker_encoder.pt"),
            device=self.device,
            compile=True,
            compile_prefill=True,
        )

    def synthesize(self, prompt: str, motivational_factor: float = 0.0):
        """
        Synthesize speech audio from a text prompt and a motivational factor.

        Args:
            prompt (str): The text to be synthesized.
            motivational_factor (float): A value between 0 and 1 representing the motivational intensity.

        Returns:
            tuple: A tuple (audio, sr) where audio is a NumPy array and sr is the sample rate.
        """
        try:
            self.logger.debug("Normalizing input text...")
            normalized_prompt = normalize_text(prompt)

            # Clamp motivational_factor to [0, 1]
            motivational_factor = max(0.0, min(motivational_factor, 1.0))
            self.logger.debug(f"Motivational factor after clamping: {motivational_factor}")
            formatted_factor = f"{float(round(motivational_factor / 0.05) * 0.05):.2f}"

            avg_speaker_emb_file = Path(self.config.average_speaker_emb_dir) / f"motivational-factor-{formatted_factor}.pt"
            if not avg_speaker_emb_file.exists():
                raise FileNotFoundError(f"Average speaker embedding file not found: {avg_speaker_emb_file}")

            self.logger.debug(f"Loading average speaker embedding from {avg_speaker_emb_file}...")
            avg_speak_emb = torch.load(str(avg_speaker_emb_file))
            avg_speak_emb = avg_speak_emb.unsqueeze(0).to(device=self.device)

            self.logger.debug("Running first stage LLM...")
            tokens = main(
                model=self.model,
                tokenizer=self.tokenizer,
                model_size=self.model_size,
                prompt=normalized_prompt,
                spk_emb=avg_speak_emb,
                top_p=torch.tensor(0.95, device=self.device, dtype=self.precision),
                guidance_scale=torch.tensor(3.0, device=self.device, dtype=self.precision),
                temperature=torch.tensor(1.0, device=self.device, dtype=self.precision),
            )

            self.logger.debug("Decoding tokens from first stage adapter...")
            text_ids, extracted_audio_ids = self.first_stage_adapter.decode([tokens])
            b_speaker_embs = avg_speak_emb.unsqueeze(0)

            self.logger.debug("Running second stage LLM for audio generation...")
            wav_files = self.llm_second_stage(
                texts=[normalized_prompt],
                encodec_tokens=[torch.tensor(extracted_audio_ids, dtype=torch.int32, device=self.device).unsqueeze(0)],
                speaker_embs=b_speaker_embs,
                batch_size=1,
                guidance_scale=None,
                top_p=None,
                top_k=200,
                temperature=1.0,
                max_new_tokens=None,
            )

            # The second stage returns a base file path; append ".wav" to get the audio file.
            enhanced_file = str(wav_files[0]) + ".wav"
            self.logger.debug("Enhancing audio with DeepFilterNet...")
            self.enhancer(enhanced_file, enhanced_file)

            self.logger.debug("Loading generated audio into memory...")
            audio, sr = librosa.load(enhanced_file, sr=None)

            # Clean up temporary file
            os.remove(enhanced_file)

            self.logger.info("Audio synthesis completed successfully.")
            return audio, sr
        except Exception as e:
            self.logger.error("Error during synthesis", exc_info=True)
            raise e

if __name__ == "__main__":
    # Import Typer only when running as a script
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        prompt: str = typer.Argument(..., help="The text to be synthesized."),
        motivational_factor: float = typer.Option(0.0, help="A value between 0 and 1 representing the motivational intensity."),
        seed: Optional[int] = typer.Option(None, help="Optional seed for inference. If not provided, a random seed will be generated."),
        intermediate_dir: str = typer.Option("motivational-speech", help="Directory for intermediate outputs."),
        output_name: str = typer.Option("generated_audio.wav", help="Name of the output audio file."),
        device: str = typer.Option("cuda:0", help="Computation device."),
        dtype: str = typer.Option("float16", help="Precision to use ('float16' or 'bfloat16')."),
        debug: bool = typer.Option(False, help="Enable debug logging."),
        average_speaker_emb_dir: str = typer.Option("average-speaker-embeddings/average-speaker-embeddings_400", help="Directory containing average speaker embeddings.")
    ):
        """
        Synthesize motivational speech audio from a text prompt.
        """
        config = MotivationalTTSConfig(
            seed=seed,
            intermediate_dir=intermediate_dir,
            device=device,
            dtype=dtype,
            debug=debug,
            average_speaker_emb_dir=average_speaker_emb_dir,
        )
        tts_model = MotivationalTTSModel(config)
        audio, sample_rate = tts_model.synthesize(prompt, motivational_factor=motivational_factor)

        import soundfile as sf
        sf.write(output_name, audio, sample_rate)
        typer.echo(f"Audio saved to {output_name} with sample rate: {sample_rate}")

    app()
