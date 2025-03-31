# Motivational Speech Synthesis

Synthesize motivational speech from text with a customizable motivational factor to control motivational prosody.

| [Preliminary Paper](https://motivationalspeechsynthesis.github.io/motivational-speech-synthesis.github.io/assets/paper.pdf) | [Project Page](https://motivationalspeechsynthesis.github.io/motivational-speech-synthesis.github.io/) | [Colab Demo](https://colab.research.google.com/github/MotivationalSpeechSynthesis/motivational-speech-synthesis/blob/main/google_colab.ipynb) |

---

So-called motivational speech has emerged as a popular audiovisual phenomenon within Western subcultures, conveying optimal strategies and principles for success through expressive, high-energy delivery. The present paper artistically explores methods for synthesizing the distinctive prosodic patterns inherent to motivational speech, while critically examining its sociocultural foundations. Drawing on recent advances in emotion-controllable text-to-speech (TTS) systems and speech emotion recognition (SER), we employ deep learning models and frameworks to replicate and analyze this genre of speech. Within our proposed architecture, we introduce a one-dimensional motivational factor derived from high-dimensional emotional speech representations, enabling the control of motivational prosody according to intensity. Situated within broader discourses on self-optimization and meritocracy, Motivational Speech Synthesis contributes to the field of emotional speech synthesis, while also prompting reflection on the societal values embedded in such mediated narratives.</p>


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

