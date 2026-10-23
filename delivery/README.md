# Standalone delivery

The directory is intentionally flat.

- `plugin.py`, `training.py`, and `metadata.py` are the reusable implementation.
- `load_example.py` loads LibriSpeech audio (about 2.74 GB), one local CAMPPlus
  checkpoint, and small slices of a prepared calibration, validation, and test
  protocol.
- `run_example.py` calls that loader, fits one bundle, loads the saved bundle,
  and produces test predictions.

No dataset, pretrained encoder, or fitted calibration bundle is copied into
this directory. The audio example needs all three locations explicitly:

```bash
python -m delivery.run_example \
  --source-root /path/to/LibriSpeech \
  --prepared-root /path/to/prepared_protocol \
  --model-directory /path/to/campplus_checkpoint \
  --output bundle.json
```
