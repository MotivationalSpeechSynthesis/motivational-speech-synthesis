# Motivational Speech Synthesis

Text-to-motivational-speech with adjustable *motivational factor* to control motivational prosody.

[Preliminary Paper](https://motivational-speech-synthesis.com/assets/motivational-speech-synthesis.pdf) | [Project Page](https://motivational-speech-synthesis.com) | [Colab Demo](https://colab.research.google.com/github/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/google_colab.ipynb)

---

*Motivational speech* has emerged as a popular audiovisual phenomenon within Western subcultures, conveying strategies and principles for individual success through expressive, high-energy delivery. The presented paper artistically explores methods for synthesizing the distinctive prosodic patterns inherent to motivational speech, while critically examining its sociocultural foundations. Drawing on recent advances in emotion-controllable text-to-speech (TTS) systems and speech emotion recognition (SER), we employ deep learning models and frameworks to replicate and analyze motivational speech. Within our proposed architecture, we introduce a one-dimensional  *motivational factor*, enabling the control of motivational prosody according to intensity. Situated within broader discourses on self-optimization and meritocracy, *Motivational Speech Synthesis* contributes to the field of emotional speech synthesis, while also prompting reflection on the societal values mediated in such narratives.
</p>


## Cloning
Use `--recurse-submodules`flag to also clone submodules
```bash
git clone --recurse-submodules git@github.com:MotivationalSpeechSynthesis/motivational-speech-synthesis.git
```

If using HTTPS rather than SSH for cloning 
```bash
git clone git@github.com:MotivationalSpeechSynthesis/motivational-speech-synthesis.git
cd motivational-speech-synthesis
git config submodule.emoknob.url https://github.com/tonychenxyz/emoknob.git
git submodule update --init --recursive
```

## Requirements

- Linux with an NVIDIA GPU (on Windows use WSL2, macOS is not supported)
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.11 and the locked dependencies

## Installation and Running

Note: Each standalone script execution recompiles the model. For repeated experiments and faster iteration, use the provided Jupyter [notebook](https://github.com/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/inference_example.ipynb).

Install the dependencies into `.venv`

```bash
uv sync
```

Run script

```bash
uv run motivationalTTS.py "Every journey begins with a single step."
```

Start Jupyter in the project environment

```bash
uv run --with jupyter jupyter lab
```

DeepFilterNet and spaCy 3.5.2 ship no wheels for Python versions newer than 3.11, so the project pins Python 3.11 in `.python-version` and uv installs it automatically. To use pip instead, export the locked versions and install them into a Python 3.11 virtual environment:

```bash
uv export --no-hashes -o requirements.txt
pip install -r requirements.txt
```

### Optional Parameters

You can customize the synthesis with the following optional arguments:

```bash
uv run motivationalTTS.py "Every journey begins with a single step." \
    --motivational-factor 0.8 \
    --seed 42 \
    --intermediate-dir "./output_audio" \
    --output-name "my_audio.wav" \
    --device "cuda:0" \
    --dtype "float16" \
    --debug \
    --average-speaker-emb-dir "average-speaker-embeddings/average-speaker-embeddings_400"
```

### Google Colab

The model can also be run with following Google Colab [example](https://colab.research.google.com/github/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/google_colab.ipynb)

Colab's own Python is too new for the dependencies, so the notebook installs them with uv and runs the model in a background process. `MotivationalTTSWorker` (in `motivationalTTSWorker.py`) offers the same `synthesize()` call as `MotivationalTTSModel` and can be used from any Python version with NumPy, as long as uv is installed (the model itself still needs Linux and an NVIDIA GPU):

```python
from motivationalTTSWorker import MotivationalTTSWorker
tts_model = MotivationalTTSWorker(seed=None, debug=False)
audio, sample_rate = tts_model.synthesize("No goal is too far away to be reached.", motivational_factor=1.0)
```

