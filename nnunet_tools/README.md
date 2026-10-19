# nnU-Net pipeline scripts

Small scripts that turn the hand-drawn masks into a trained 2D nnU-Net model
and back into starting masks for the GUI. `predict_worker.py` is started by the
GUI's **Use model** checkbox; the other scripts are run by hand from the
repository root.

The training environment lives outside the GUI's environment, because nnU-Net
pulls its own torch, SimpleITK and friends:

```bash
python -m venv .venv-nnunet
.venv-nnunet/bin/python -m pip install nnunetv2==2.8.1
```

## Order of operations

| step | script | what it writes |
| --- | --- | --- |
| 1 | `build_dataset.py` | `nnUNet/nnUNet_raw/Dataset503_BBDBCortex/` — one PNG pair per section that has a mask, plus `pending_pngs/` for sections that do not |
| 2 | `make_synthetic.py` | `nnUNet/nnUNet_raw/Dataset504_BBDBCortexSynth/` — the real pairs plus N synthetic variants each |
| 3 | `make_grouped_splits.py` | replaces `splits_final.json` so all variants of a section share a fold |
| 4 | `nnUNetv2_plan_and_preprocess -d 504 --verify_dataset_integrity` | plans and preprocessed cases |
| 5 | `nnUNetv2_train 504 2d 0 -tr nnUNetTrainer_333epochs` | checkpoints under `nnUNet/nnUNet_results/` |
| 6 | `nnUNetv2_predict ...` | predicted masks for unseen sections |
| 7 | `export_predictions.py` | the same masks in `<group>/<base>.png` layout, ready for `--init-dir` |

`nnUNetTrainer_333epochs.py` is a copy of the file that must sit inside the
installed package (`nnunetv2/training/nnUNetTrainer/variants/training_length/`)
for `-tr nnUNetTrainer_333epochs` to resolve. The packaged 504 checkpoint also
records that trainer name, so inference needs the class too: copy it into the
installed package before starting the worker. The stock `nnUNetTrainer` name is
not enough (the packaged `HOWTO.txt` is wrong on this point).

## Inference worker (used by the GUI)

`predict_worker.py` loads the model once and then answers one JSON request per
line. The GUI starts it with `.venv-nnunet/bin/python` and caches the masks in
`segmentations/UNet_output/<group>/<base>.png`.

```bash
printf '{"cmd":"quit"}\n' | .venv-nnunet/bin/python nnunet_tools/predict_worker.py \
    --model-dir bbdb_model_504 --device auto
```

`--model-dir` accepts the extracted `bbdb_model_504/` package or the trainer
folder that contains `fold_0/`.

## Why synthetic images

31 hand-drawn masks are not enough to generalise. `make_synthetic.py` follows
the recipe in the project README: build a label map from the mask and an Otsu
tissue mask, optionally split the cortex into pseudo-layers, warp it, paint
random intensities with the cortex randomly darker or brighter than its
surroundings, add acquisition artefacts, and keep the deformed mask as target.

## Splits matter

With 100 variants per section, a plain random split leaks: `caseA__syn003`
trains, `caseA__syn057` validates. `make_grouped_splits.py` groups by the part
of the case name before `__syn`, so validation always contains sections the
model has never seen.

## Everything is regenerable

`nnUNet/` is ignored by git. The whole tree (raw data, preprocessed cases,
checkpoints) can be rebuilt from `segmentations/`, `csv/` and
`brain_biodiversity_bank/` by re-running the table above.
