# Motivational Speech Synthesis

Generate motivational speech audio from text inputs with customizable parameters to enhance the motivational tone.

## Requirements

- Python 3.9
- Linux OS recommended (Windows support expected, macOS currently unsupported)

## Installation

This project utilizes `uv` for package management and virtual environment handling.

### Using uv (Recommended)

Create and activate the virtual environment:

```bash
uv venv
uv pip install -r requirements.txt
```

### Using pip

Alternatively, you can use `pip`:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Run the synthesis script directly from the command line:

```bash
uv run motivationalTTS.py "Every journey begins with a single step."
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
    --average-speaker-emb-dir "average_speaker_embeddings/average-speaker-embeddings_400"
```

Note: Each standalone script execution recompiles the model. For repeated experiments and faster iteration, use the provided Jupyter notebook.

## Interactive Experimentation

For easier experimentation, run the provided Jupyter notebook:

```bash
uv run jupyter-notebook
```

or, using pip:

```bash
jupyter-notebook
```

You can also initiate the virtual environment setup directly by executing the first cell within the notebook itself.