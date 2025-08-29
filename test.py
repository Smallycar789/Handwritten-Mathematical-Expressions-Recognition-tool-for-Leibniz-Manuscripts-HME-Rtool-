from PIL import Image
from processor import recognize


def main():
    image_path = r"/home/qian-z/stage_qianze/HME-Rtool/data/test/images/crop_20250821-132354.png"
    version = "28"

    # Load PNG as PIL image
    img = Image.open(image_path)

    # Run recognition
    print(f"Running recognition on {image_path} with model version {version} ...")
    pred_text = recognize(img, version=version)

    # Print final result
    print("\n=== Recognition Result ===")
    print(pred_text)


if __name__ == "__main__":
    main()
'''

import os
import typer
from comer.datamodule.datamodule import LHDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything
import sys

seed_everything(7)

def main(version: str):
    # Load checkpoint path from logs
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1, f"Expected one checkpoint file, found {len(fnames)} in {ckp_folder}"
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Testing with checkpoint: {ckp_path}")

    # Initialize trainer and datamodule
    trainer = Trainer(logger=False, gpus=0)  # or accelerator="gpu", devices=1 for newer PL
    dm = LHDatamodule(eval_batch_size=1)  # loads 'test' dataset in setup(stage="test")

    # Load model and test
    model = LitCoMER.load_from_checkpoint(ckp_path)
    trainer.test(model)

if __name__ == "__main__":
    typer.run(main)
'''