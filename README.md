# Motivational Speech Synthesis

Synthesize motivational speech from text with a customizable motivational factor to control motivational prosody.

| [Preliminary Paper](https://motivationalspeechsynthesis.github.io/motivational-speech-synthesis.github.io/assets/paper.pdf) | [Project Page](https://motivationalspeechsynthesis.github.io/motivational-speech-synthesis.github.io/) | [Colab Demo](https://colab.research.google.com/github/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/google_colab.ipynb) | [Dataset](TODO) |

## Cloning
Use `--recurse-submodules`flag to also clone submodules
```bash
git clone --recurse-submodules git@github.com:MotivationalSpeechSynthesis/motivational-speech-synthesis.git
```

## Requirements

- Linux OS recommended (Windows support expected but not tested, macOS currently unsupported)

## Installation and Running

Note: Each standalone script execution recompiles the model. For repeated experiments and faster iteration, use the provided Jupyter [notebook](https://github.com/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/inference_example.ipynb).

### Using uv

Run script

```bash
uv run motivationalTTS.py "Every journey begins with a single step."
```

Virtual env for jupyter-notebook:

```bash
uv venv
```

Start jupyter-notebook

```bash
uv run jupyter-notebook
```

### Using pip

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run script

```bash
python motivationalTTS.py "Every journey begins with a single step."
```

Start jupyter-notebook

```bash
uv run jupyter-notebook
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

