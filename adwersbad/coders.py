from io import BytesIO

import numpy as np
import torch
from torchvision import tv_tensors
from PIL import Image

def encode_image(image: torch.Tensor, format=".jpg", **kwargs) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def decode_image(encoded_image: bytes) -> torch.Tensor:
    pi = np.array(Image.open(BytesIO(encoded_image)))
    # Wrap in Image, s.t. the correct transformation are applied
    return tv_tensors.Image(
        torch.from_numpy(pi).permute(2, 0, 1).to(torch.float32) / 255.0
    )


def encode_image_label(label: torch.Tensor, format=".png", **kwargs) -> bytes:
    pi = label.cpu().numpy()
    buffer = BytesIO()
    Image.fromarray(pi).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def decode_image_label(encoded_image: bytes) -> torch.Tensor:
    pi = np.array(Image.open(BytesIO(encoded_image)))
    # Wrap in Mask, s.t. the correct transformation are applied
    return tv_tensors.Mask(torch.from_numpy(pi).squeeze(-1).to(torch.long))


def encode_lidar(lidar: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, lidar)
    return buffer.getvalue()



def decode_lidar(encoded_lidar: bytes) -> torch.Tensor:
    pi = np.load(BytesIO(encoded_lidar))
    return torch.from_numpy(pi).to(torch.float32)


def encode_lidar_label(label: np.ndarray, format="PNG", **kwargs) -> bytes:
    buffer = BytesIO()
    Image.fromarray(label).save(buffer, format=format)
    return buffer.getvalue()



def decode_lidar_label(encoded_label: bytes) -> torch.Tensor:
    pi = np.array(Image.open(BytesIO(encoded_label)))
    # TODO: could wrap in tv_tensors.Mask(...), but there's no Lidar equivalent
    return torch.from_numpy(pi).squeeze(-1).to(torch.long)


def encode_probabilities(probs: torch.Tensor) -> bytes:
    # Save tensor with compression via numpy.savez_compressed
    pi = probs.mul(255.0).to(torch.uint8).cpu().numpy()
    buffer = BytesIO()
    np.savez_compressed(buffer, probs=pi)
    return buffer.getvalue()


def decode_probabilities(encoded_probs: bytes) -> torch.Tensor:
    probs = np.load(BytesIO(encoded_probs))["probs"]
    probs = torch.from_numpy(probs).div(255.0).to(torch.float32)
    return probs / probs.sum(axis=0)


def decode_lidar_proj(encoded_projections: bytes) -> np.ndarray:
    proj = np.load(BytesIO(encoded_projections))
    return proj
