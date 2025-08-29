import os
import torch
from PIL import Image
from comer.lit_comer import LitCoMER
from comer.datamodule.vocab import vocab
from comer.datamodule.transforms import ScaleToLimitRange
from torchvision.transforms import ToTensor

torch.set_num_threads(4)

# Transform configuration, adjust if needed to match training
_transform = ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256)

_model = None
_device = None

def preprocess_image(pil_img: Image.Image):
    """Preprocess input PIL image for model inference."""
    img = pil_img.convert("L")
    img = ToTensor()(img)
    mask = torch.zeros_like(img, dtype=torch.bool)
    return img.unsqueeze(0), mask

def load_model(version="28"):
    """Load and prepare the model only once."""
    global _model, _device

    if _model is not None:
        print("Model already loaded.")
        return _model, _device

    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1, f"Expected exactly one checkpoint in {ckp_folder}, found {len(fnames)}"
    ckp_path = os.path.join(ckp_folder, fnames[0])
    
    _model = LitCoMER.load_from_checkpoint(ckp_path)
    _model.eval()
    _model.freeze()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    print(f"Load the model version_{version}") 
    return _model, _device

def recognize(image_crop: Image.Image, version="28"):
    
    """
    Recognize text from a cropped PIL image using the loaded model.
    """
    model, device = load_model(version)
    img_tensor, mask = preprocess_image(image_crop)
    img_tensor = img_tensor.to(device)  # [1, 1, H, W], FloatTensor
    mask = mask.to(device)              # [1, H, W], LongTensor
    print('image shape;', img_tensor.shape)
    print('mask shape:', mask.shape)
    print('img_tensor min:', img_tensor.min().item(), 'max:', img_tensor.max().item(), 'mean:', img_tensor.mean().item())
    print('Begin recognition')
    hyps = model.approximate_joint_search(img_tensor, mask)
    print('Finish the recognition')
    pred_indices = hyps[0].seq
    pred_text = vocab.indices2label(pred_indices)
    
    return pred_text

