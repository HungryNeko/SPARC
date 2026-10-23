# In-browser demo

`../demo.html` compares two recordings entirely in the browser with ONNX Runtime Web 1.30.0.

| File | What it is | Source and license |
|---|---|---|
| `models/campplus.onnx` | CAMPPlus speaker encoder with the Kaldi fbank-80 front end inside the graph; weights stored as float16, computed in float32 | 3D-Speaker `iic/speech_campplus_sv_zh-cn_16k-common`, Apache License 2.0 |
| `models/lid.onnx` | SpeechBrain VoxLingua107 ECAPA language identification with its fbank front end inside the graph; weight-only 8-bit | `speechbrain/lang-id-voxlingua107-ecapa`, Apache License 2.0 |
| `models/silero_vad.onnx` | Silero VAD, unchanged | `snakers4/silero-vad` 6.2.1, MIT License |
| `bundle.json` | SPARC calibration fitted for this demo with `delivery.Plugin.fit` on LibriSpeech dev-clean (calibration) and dev-other (validation) | this repository |
| `evaluation.json` | Test metrics, reliability bins, the example draw and browser reference values | this repository |
| `examples/*.flac` | Six LibriSpeech test utterances | LibriSpeech (Panayotov et al., 2015), CC BY 4.0 |
| `lid_labels.json` | VoxLingua107 label order of `lid.onnx` | `speechbrain/lang-id-voxlingua107-ecapa` |

The model files were converted to ONNX; apart from the weight storage formats above, they are not modified.
Apache-2.0 and MIT license texts: <https://www.apache.org/licenses/LICENSE-2.0>, <https://opensource.org/license/mit>.

Pipeline: 16 kHz mono audio (first 30 s) → Silero VAD speech duration → CAMPPlus embedding → cosine score;
VoxLingua107 probabilities and embedding → the 15 SPARC context features → SPARC log-likelihood ratio.
In the browser the cosine score matches the Python reference within 3e-6 and the SPARC LLR within 0.01.

The calibration is specific to English read speech. Probabilities for other languages or recording conditions may be unreliable.
