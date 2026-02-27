import os
from transformers import AutoProcessor, LlavaForConditionalGeneration # llava
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,LlavaNextImageProcessor,Qwen2VLImageProcessor
import torch
from PIL import Image
import requests
from io import BytesIO
from cog import BasePredictor, Input, Path, ConcatenateIterator
import json
import numpy as np
import tqdm
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
from torchvision.transforms import transforms
import torch.nn.functional as F



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    scaled_image = (image_tensor * 255).clamp(0, 255).to(torch.uint8)
    return scaled_image

def is_scaled_image(image: torch.Tensor)-> bool:
    if image.dtype == torch.uint8:
        return False  
    
    return torch.min(image) >= 0 and torch.max(image) <= 1


def rescale_image(image: torch.Tensor)-> torch.Tensor:
    if is_scaled_image(image):
        return image
    else:
        return image / 255.0


def get_image_size( 
    image: Union[np.ndarray, torch.Tensor])-> Tuple[int,int]:
    if len(image.shape) == 3:
        if image.shape[0] == 3:  
            height = image.shape[1]
            width = image.shape[2]
            return height, width
        elif image.shape[2] == 3:  
            height = image.shape[0]
            width = image.shape[1]
            return height, width
    elif len(image.shape) == 4:
        if image.shape[1] == 3:  
            height = image.shape[2]
            width = image.shape[3]
            return height, width
        elif image.shape[3] == 3:  
            height = image.shape[1]
            width = image.shape[2]
            return height, width
            
def convert_tensor_shape(image_tensor: torch.Tensor)-> torch.Tensor:
    if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
        return image_tensor

    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError("输入的张量必须是形状为 [3, H, W] 的张量")
    return image_tensor.permute(1, 2, 0)

def convert_tensor_shape_back(image_tensor: torch.Tensor)-> torch.Tensor:
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        return image_tensor
    
    if image_tensor.dim() != 3 or image_tensor.shape[2] != 3:
        raise ValueError("输入的张量必须是形状为 [H, W, 3] 的张量")
    return image_tensor.permute(2, 0, 1)

def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit

def _get_patch_output_size(image, target_resolution):
    original_height, original_width = get_image_size(image)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width


def pad_for_patching(
    image: torch.Tensor, target_resolution: tuple
) -> torch.Tensor:
    """
    Pad an image to a target resolution while maintaining aspect ratio.
    """
    target_height, target_width = target_resolution
    new_height, new_width = _get_patch_output_size(image, target_resolution)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_image = pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))

    return padded_image

def pad (
        image: torch.Tensor,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        constant_values: Union[float, Iterable[float]] = 0.0,
    )-> torch.Tensor:
        if isinstance(padding[0], tuple):
            padded_image = F.pad(image, padding[0], mode="constant", value=constant_values)
        else:
            padded_image = F.pad(image, padding, mode="constant", value=constant_values)
        return padded_image


def resize(
        image: torch.Tensor,
        size: Tuple[int, int],
    )-> torch.Tensor:
    
    resize_transform = transforms.Resize(size)  # 创建 resize 转换
    resized_image_tensor = resize_transform(image)  # 进行 resize
    return resized_image_tensor


def resize_for_patching(
        image: torch.Tensor, target_resolution: tuple
    ) -> torch.Tensor:

        new_height, new_width = _get_patch_output_size(image, target_resolution)

        # Resize the image
        resized_image = resize(image, (new_height, new_width))

        return resized_image

def divide_to_patches(
        image: torch.Tensor,
        patch_size: int,
    )->List[torch.Tensor]:
    patches = []
    height, width = get_image_size(image)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches



def normalize(
    image: torch.Tensor,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    num_channels: int = 3,
    **kwargs,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a torch tensor")
    
    if not image.is_floating_point():
        image = image.to(torch.float32)

    image = convert_tensor_shape_back(image)
    device = image.device

    # mean = [mean] * num_channels
    mean = torch.tensor(mean, dtype=image.dtype).unsqueeze(-1).unsqueeze(-1).to(device)


    # std = [std] * num_channels
    std = torch.tensor(std, dtype=image.dtype).unsqueeze(-1).unsqueeze(-1).to(device)

    image = (image - mean) / std

    image = convert_tensor_shape(image)

    return image

import torch
from typing import List, Optional, Union

def _pad_for_batching(

    pixel_values: List[torch.Tensor],

    ) -> List[torch.Tensor]:
    max_patch = max(len(x) for x in pixel_values)
    
    pixel_values = [
        pad(
            image,
            padding=(0, max_patch - image.shape[0]),  
        )
        for image in pixel_values
    ]

    return pixel_values

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def get_image_patches(
        image: torch.Tensor,
        grid_pinpoints: List[str],
        size: Tuple[int, int],
        patch_size: int,
        
    )->List[torch.Tensor]:
    import PIL.Image
    PILImageResampling = PIL.Image.Resampling

    possible_resolutions = grid_pinpoints
    resample = PILImageResampling
    image_size = get_image_size(image)
    best_resolution = select_best_resolution(image_size, possible_resolutions)
    resized_image = resize_for_patching(image, best_resolution)
    padded_image = pad_for_patching(image=resized_image, target_resolution=best_resolution)

    patches = divide_to_patches(padded_image, patch_size=patch_size)
    patches = [convert_tensor_shape(patch) for patch in patches]

    resized_original_image = resize(image, size = (336, 336))
    resized_original_image = convert_tensor_shape(resized_original_image)

    image_patches = [resized_original_image] + patches

    return image_patches


def _preprocess(
    images,
    do_resize: bool = None,
    size: Dict[str, int] = None,
    do_center_crop: bool = None,
    crop_size: int = None,
    do_rescale: bool = None,
    rescale_factor: float = None,
    do_normalize: bool = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    **kwargs,
)-> torch.Tensor:
    images = [images]
    processed_images = []
    for image in images:

        height, width = get_image_size(image)

        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=14 * 2,
                min_pixels=3136,
                max_pixels=12845056,
                )
            image = resize(image, (resized_height, resized_width))
        # if do_center_crop:
        #     image = center_crop(image, crop_size)
        if do_rescale:
            image = rescale_image(image)
        if do_normalize:
            image = normalize(image, image_mean, image_std)
        
        processed_images.append(convert_tensor_shape_back(image))
    

    if len(processed_images) == 1:
        processed_images = processed_images[0]

    patches = processed_images.unsqueeze(0)

    if patches.shape[0] == 1:
        patches = patches.repeat(2, 1, 1, 1)
    
    channel = patches.shape[1]

    grid_t = patches.shape[0] // 2
    grid_h, grid_w = resized_height // 14, resized_width // 14

    patches = patches.view(
    grid_t,
    2,
    channel,
    grid_h // 2,
    2,
    14,
    grid_w // 2,
    2,
    14,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * 2 * 14 * 14
    )


    return flatten_patches, (grid_t, grid_h, grid_w)

    


def preprocess(
    images,
    do_resize: bool = True,
    do_center_crop: bool = True,
    do_rescale: bool = True,
    do_normalize: bool = True,
    


)-> Image.Image:
    image = images
    size = {'shortest_edge': 336}
    image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    resample = 3
    crop_size = {'height': 336, 'width': 336}
    rescale_factor = 0.00392156862745098
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]

    new_images = []


    image_size = get_image_size(image)
    # image_patches = get_image_patches(image, image_grid_pinpoints, size, crop_size['height'])

    pixel_values, vision_grid_thws = [], []
    patches, image_grid_thw = _preprocess(
    # image_patches,
    image,
    do_resize=do_resize,
    size=size,
    do_center_crop=do_center_crop,
    crop_size=crop_size,
    do_rescale=do_rescale,
    rescale_factor=rescale_factor,
    do_normalize=do_normalize,
    image_mean=image_mean,
    image_std=image_std,
    )


    pixel_values = patches
    vision_grid_thws.append(image_grid_thw)


    vision_grid_thws = torch.tensor(vision_grid_thws)
    data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

    return data['pixel_values']



    





