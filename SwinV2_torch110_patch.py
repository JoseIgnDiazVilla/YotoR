"""
WARNING 
This implementation of pytorch swinv2 was made to solve a compatibility problem 
in an university server to tran the model with pytorch 1.10.

If for some reazon you are reading this: DON'T USE THIS and just update pytorch to version >= 1.13 
and import the swinv2 models directly 
"""

from PIL import Image

# See https://pillow.readthedocs.io/en/stable/releasenotes/9.1.0.html#deprecations
# TODO: Remove this file once PIL minimal version is >= 9.1

if hasattr(Image, "Resampling"):
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
    LINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST

    AFFINE = Image.Transform.AFFINE
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.Transform.PERSPECTIVE
else:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    LINEAR = Image.LINEAR

    AFFINE = Image.AFFINE
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    PERSPECTIVE = Image.PERSPECTIVE


    import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad


def F_t__is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def F_t__assert_image_tensor(img: Tensor) -> None:
    if not F_t__is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def F_t__assert_threshold(img: Tensor, threshold: float) -> None:
    bound = 1 if img.is_floating_point() else 255
    if threshold > bound:
        raise TypeError("Threshold should be less than bound of img.")


def F_t_get_dimensions(img: Tensor) -> List[int]:
    F_t__assert_image_tensor(img)
    channels = 1 if img.ndim == 2 else img.shape[-3]
    height, width = img.shape[-2:]
    return [channels, height, width]


def F_t_get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    F_t__assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


def F_t_get_image_num_channels(img: Tensor) -> int:
    F_t__assert_image_tensor(img)
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError(f"Input ndim should be 2 or more. Got {img.ndim}")


def F_t__max_value(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 255
    elif dtype == torch.int8:
        return 127
    elif dtype == torch.int16:
        return 32767
    elif dtype == torch.int32:
        return 2147483647
    elif dtype == torch.int64:
        return 9223372036854775807
    else:
        return 1


def F_t__assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = get_dimensions(img)[0]
    if c not in permitted:
        raise TypeError(f"Input image tensor permitted channel values are {permitted}, but found {c}")


def F_t_convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():

        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=dtype).is_floating_point():
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and dtype == torch.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = float(F_t__max_value(dtype))
        result = image.mul(max_val + 1.0 - eps)
        return result.to(dtype)
    else:
        input_max = float(F_t__max_value(image.dtype))

        # int to float
        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=dtype).is_floating_point():
            image = image.to(dtype)
            return image / input_max

        output_max = float(F_t__max_value(dtype))

        # int to int
        if input_max > output_max:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image // factor can produce different results
            factor = int((input_max + 1) // (output_max + 1))
            image = torch.div(image, factor, rounding_mode="floor")
            return image.to(dtype)
        else:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image * factor can produce different results
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(dtype)
            return image * factor


def F_t_vflip(img: Tensor) -> Tensor:
    F_t__assert_image_tensor(img)

    return img.flip(-2)


def F_t_hflip(img: Tensor) -> Tensor:
    F_t__assert_image_tensor(img)

    return img.flip(-1)


def F_t_crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    F_t__assert_image_tensor(img)

    _, h, w = get_dimensions(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)]
        return pad(img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0)
    return img[..., top:bottom, left:right]


def F_t_rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    F_t__assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def F_t_adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")

    F_t__assert_image_tensor(img)

    F_t__assert_channels(img, [1, 3])

    return F_t__blend(img, torch.zeros_like(img), brightness_factor)


def F_t_adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")

    F_t__assert_image_tensor(img)

    F_t__assert_channels(img, [3, 1])
    c = F_t_get_dimensions(img)[0]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    if c == 3:
        mean = torch.mean(F_t_rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    else:
        mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)

    return F_t__blend(img, mean, contrast_factor)


def F_t_adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor image")

    F_t__assert_image_tensor(img)

    F_t__assert_channels(img, [1, 3])
    if F_t_get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = F_t__rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = F_t__hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def F_t_adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    if saturation_factor < 0:
        raise ValueError(f"saturation_factor ({saturation_factor}) is not non-negative.")

    F_t__assert_image_tensor(img)

    F_t__assert_channels(img, [1, 3])

    if F_t_get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    return F_t__blend(img, F_t_rgb_to_grayscale(img), saturation_factor)


def F_t_adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input img should be a Tensor.")

    F_t__assert_channels(img, [1, 3])

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    result = img
    dtype = img.dtype
    if not torch.is_floating_point(img):
        result = F_t_convert_image_dtype(result, torch.float32)

    result = (gain * result**gamma).clamp(0, 1)

    result = F_t_convert_image_dtype(result, dtype)
    return result


def F_t__blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def F_t__rgb2hsv(img: Tensor) -> Tensor:
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def F_t__hsv2rgb(img: Tensor) -> Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


def F_t__pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + _x_indices + right_indices, device=img.device)

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + _y_indices + bottom_indices, device=img.device)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def F_t__parse_pad_padding(padding: Union[int, List[int]]) -> List[int]:
    if isinstance(padding, int):
        if torch.jit.is_scripting():
            # This maybe unreachable
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    return [pad_left, pad_right, pad_top, pad_bottom]


def F_t_pad(
    img: Tensor, padding: Union[int, List[int]], fill: Union[int, float] = 0, padding_mode: str = "constant"
) -> Tensor:
    F_t__assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list):
        # TODO: Jit is failing on loading this op when scripted and saved
        # https://github.com/pytorch/pytorch/issues/81100
        if len(padding) not in [1, 2, 4]:
            raise ValueError(
                f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple"
            )

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    p = F_t__parse_pad_padding(padding)

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return F_t__pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (torch.float32, torch.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = img.to(torch.float32)

    if padding_mode in ("reflect", "replicate"):
        img = torch_pad(img, p, mode=padding_mode)
    else:
        img = torch_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = img.to(out_dtype)

    return img


def F_t_resize(
    img: Tensor,
    size: List[int],
    interpolation: str = "bilinear",
    antialias: Optional[bool] = None,
) -> Tensor:
    F_t__assert_image_tensor(img)

    if isinstance(size, tuple):
        size = list(size)

    if antialias is None:
        antialias = False

    if antialias and interpolation not in ["bilinear", "bicubic"]:
        raise ValueError("Antialias option is supported for bilinear and bicubic interpolation modes only")

    img, need_cast, need_squeeze, out_dtype = F_t__cast_squeeze_in(img, [torch.float32, torch.float64])

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    img = interpolate(img, size=size, mode=interpolation, align_corners=align_corners, antialias=antialias)

    if interpolation == "bicubic" and out_dtype == torch.uint8:
        img = img.clamp(min=0, max=255)

    img = F_t__cast_squeeze_out(img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype)

    return img


def F_t__assert_grid_transform_inputs(
    img: Tensor,
    matrix: Optional[List[float]],
    interpolation: str,
    fill: Optional[List[float]],
    supported_interpolation_modes: List[str],
    coeffs: Optional[List[float]] = None,
) -> None:

    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor")

    F_t__assert_image_tensor(img)

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None and not isinstance(fill, (int, float, tuple, list)):
        warnings.warn("Argument fill should be either int, float, tuple or list")

    # Check fill
    num_channels = F_t_get_dimensions(img)[0]
    if isinstance(fill, (tuple, list)) and (len(fill) > 1 and len(fill) != num_channels):
        msg = (
            "The number of elements in 'fill' cannot broadcast to match the number of "
            "channels of the image ({} != {})"
        )
        raise ValueError(msg.format(len(fill), num_channels))

    if interpolation not in supported_interpolation_modes:
        raise ValueError(f"Interpolation mode '{interpolation}' is unsupported with Tensor input")


def F_t__cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def F_t__cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def F_t__apply_grid_transform(img: Tensor, grid: Tensor, mode: str, fill: Optional[List[float]]) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = F_t__cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, dummy), dim=1)

    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = F_t__cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def F_t__gen_affine_grid(
    theta: Tensor,
    w: int,
    h: int,
    ow: int,
    oh: int,
) -> Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def F_t_affine(
    img: Tensor, matrix: List[float], interpolation: str = "nearest", fill: Optional[List[float]] = None
) -> Tensor:
    F_t__assert_grid_transform_inputs(img, matrix, interpolation, fill, ["nearest", "bilinear"])

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = F_t__gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    return F_t__apply_grid_transform(img, grid, interpolation, fill=fill)


def _compute_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:

    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    # Points are shifted due to affine matrix torch convention about
    # the center point. Center is (0, 0) for image center pivot point (w * 0.5, h * 0.5)
    pts = torch.tensor(
        [
            [-0.5 * w, -0.5 * h, 1.0],
            [-0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, -0.5 * h, 1.0],
        ]
    )
    theta = torch.tensor(matrix, dtype=torch.float).view(2, 3)
    new_pts = torch.matmul(pts, theta.T)
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # shift points to [0, w] and [0, h] interval to match PIL results
    min_vals += torch.tensor((w * 0.5, h * 0.5))
    max_vals += torch.tensor((w * 0.5, h * 0.5))

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = torch.ceil((max_vals / tol).trunc_() * tol)
    cmin = torch.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])  # w, h


def F_t_rotate(
    img: Tensor,
    matrix: List[float],
    interpolation: str = "nearest",
    expand: bool = False,
    fill: Optional[List[float]] = None,
) -> Tensor:
    F_t__assert_grid_transform_inputs(img, matrix, interpolation, fill, ["nearest", "bilinear"])
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = F_t__gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)

    return F_t__apply_grid_transform(img, grid, interpolation, fill=fill)


def F_t__perspective_grid(coeffs: List[float], ow: int, oh: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]], dtype=dtype, device=device
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]], dtype=dtype, device=device)

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


def F_t_perspective(
    img: Tensor, perspective_coeffs: List[float], interpolation: str = "bilinear", fill: Optional[List[float]] = None
) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor.")

    F_t__assert_image_tensor(img)

    F_t__assert_grid_transform_inputs(
        img,
        matrix=None,
        interpolation=interpolation,
        fill=fill,
        supported_interpolation_modes=["nearest", "bilinear"],
        coeffs=perspective_coeffs,
    )

    ow, oh = img.shape[-1], img.shape[-2]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    grid = F_t__perspective_grid(perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=img.device)
    return F_t__apply_grid_transform(img, grid, interpolation, fill=fill)


def F_t__get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def F_t__get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = F_t__get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = F_t__get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def F_t_gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    F_t__assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = F_t__get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = F_t__cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = F_t__cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def F_t_invert(img: Tensor) -> Tensor:

    F_t__assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    F_t__assert_channels(img, [1, 3])

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def F_t_posterize(img: Tensor, bits: int) -> Tensor:

    F_t__assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    if img.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {img.dtype}")

    F_t__assert_channels(img, [1, 3])
    mask = -int(2 ** (8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return img & mask


def F_t_solarize(img: Tensor, threshold: float) -> Tensor:

    F_t__assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    F_t__assert_channels(img, [1, 3])

    F_t__assert_threshold(img, threshold)

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)


def F_t__blurred_degenerate_image(img: Tensor) -> Tensor:
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    kernel = torch.ones((3, 3), dtype=dtype, device=img.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    result_tmp, need_cast, need_squeeze, out_dtype = F_t__cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )
    result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
    result_tmp = F_t__cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)

    result = img.clone()
    result[..., 1:-1, 1:-1] = result_tmp

    return result


def F_t_adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")

    F_t__assert_image_tensor(img)

    F_t__assert_channels(img, [1, 3])

    if img.size(-1) <= 2 or img.size(-2) <= 2:
        return img

    return F_t__blend(img, F_t__blurred_degenerate_image(img), sharpness_factor)


def F_t_autocontrast(img: Tensor) -> Tensor:

    F_t__assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    F_t__assert_channels(img, [1, 3])

    bound = 1.0 if img.is_floating_point() else 255.0
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    minimum = img.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum = img.amax(dim=(-2, -1), keepdim=True).to(dtype)
    scale = bound / (maximum - minimum)
    eq_idxs = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs] = 1

    return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)


def F_t__scale_channel(img_chan: Tensor) -> Tensor:
    # TODO: we should expect bincount to always be faster than histc, but this
    # isn't always the case. Once
    # https://github.com/pytorch/pytorch/issues/53194 is fixed, remove the if
    # block and only use bincount.
    if img_chan.is_cuda:
        hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)
    else:
        hist = torch.bincount(img_chan.view(-1), minlength=256)

    nonzero_hist = hist[hist != 0]
    step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode="floor")
    if step == 0:
        return img_chan

    lut = torch.div(torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode="floor"), step, rounding_mode="floor")
    lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)

    return lut[img_chan.to(torch.int64)].to(torch.uint8)


def F_t__equalize_single_image(img: Tensor) -> Tensor:
    return torch.stack([F_t__scale_channel(img[c]) for c in range(img.size(0))])


def F_t_equalize(img: Tensor) -> Tensor:

    F_t__assert_image_tensor(img)

    if not (3 <= img.ndim <= 4):
        raise TypeError(f"Input image tensor should have 3 or 4 dimensions, but found {img.ndim}")
    if img.dtype != torch.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {img.dtype}")

    F_t__assert_channels(img, [1, 3])

    if img.ndim == 3:
        return F_t__equalize_single_image(img)

    return torch.stack([F_t__equalize_single_image(x) for x in img])


def F_t_normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    F_t__assert_image_tensor(tensor)

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def F_t_erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
    F_t__assert_image_tensor(img)

    if not inplace:
        img = img.clone()

    img[..., i : i + h, j : j + w] = v
    return img


def F_t__create_identity_grid(size: List[int]) -> Tensor:
    hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size]
    grid_y, grid_x = torch.meshgrid(hw_space, indexing="ij")
    return torch.stack([grid_x, grid_y], -1).unsqueeze(0)  # 1 x H x W x 2


def F_t_elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: str = "bilinear",
    fill: Optional[List[float]] = None,
) -> Tensor:

    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    size = list(img.shape[-2:])
    displacement = displacement.to(img.device)

    identity_grid = F_t__create_identity_grid(size)
    grid = identity_grid.to(img.device) + displacement
    return F_t__apply_grid_transform(img, grid, interpolation, fill)


import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from typing_extensions import Literal

try:
    import accimage
except ImportError:
    accimage = None


@torch.jit.unused
def F_pil__is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@torch.jit.unused
def F_pil_get_dimensions(img: Any) -> List[int]:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]
    raise TypeError(f"Unexpected type {type(img)}")


@torch.jit.unused
def F_pil_get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return list(img.size)
    raise TypeError(f"Unexpected type {type(img)}")


@torch.jit.unused
def F_pil_get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    raise TypeError(f"Unexpected type {type(img)}")


@torch.jit.unused
def F_pil_hflip(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    return img.transpose(FLIP_LEFT_RIGHT)


@torch.jit.unused
def F_pil_vflip(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    return img.transpose(FLIP_TOP_BOTTOM)


@torch.jit.unused
def F_pil_adjust_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


@torch.jit.unused
def F_pil_adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


@torch.jit.unused
def F_pil_adjust_saturation(img: Image.Image, saturation_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


@torch.jit.unused
def F_pil_adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


@torch.jit.unused
def F_pil_adjust_gamma(
    img: Image.Image,
    gamma: float,
    gain: float = 1.0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    input_mode = img.mode
    img = img.convert("RGB")
    gamma_map = [int((255 + 1 - 1e-3) * gain * pow(ele / 255.0, gamma)) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


@torch.jit.unused
def F_pil_pad(
    img: Image.Image,
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError(f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple")

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if padding_mode == "constant":
        opts = F_pil__parse_fill(fill, img, name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == "P":
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)


@torch.jit.unused
def F_pil_crop(
    img: Image.Image,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    return img.crop((left, top, left + width, top + height))


@torch.jit.unused
def F_pil_resize(
    img: Image.Image,
    size: Union[List[int], int],
    interpolation: int = BILINEAR,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    if not (isinstance(size, list) and len(size) == 2):
        raise TypeError(f"Got inappropriate size arg: {size}")

    return img.resize(tuple(size[::-1]), interpolation)


@torch.jit.unused
def F_pil__parse_fill(
    fill: Optional[Union[float, List[float], Tuple[float, ...]]],
    img: Image.Image,
    name: str = "fillcolor",
) -> Dict[str, Optional[Union[float, List[float], Tuple[float, ...]]]]:

    # Process fill color for affine transforms
    num_channels = get_image_num_channels(img)
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_channels > 1:
        fill = tuple([fill] * num_channels)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_channels:
            msg = "The number of elements in 'fill' does not match the number of channels of the image ({} != {})"
            raise ValueError(msg.format(len(fill), num_channels))

        fill = tuple(fill)

    if img.mode != "F":
        if isinstance(fill, (list, tuple)):
            fill = tuple(int(x) for x in fill)
        else:
            fill = int(fill)

    return {name: fill}


@torch.jit.unused
def F_pil_affine(
    img: Image.Image,
    matrix: List[float],
    interpolation: int = NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    output_size = img.size
    opts = F_pil__parse_fill(fill, img)
    return img.transform(output_size, AFFINE, matrix, interpolation, **opts)


@torch.jit.unused
def F_pil_rotate(
    img: Image.Image,
    angle: float,
    interpolation: int = NEAREST,
    expand: bool = False,
    center: Optional[Tuple[int, int]] = None,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    opts = F_pil__parse_fill(fill, img)
    return img.rotate(angle, interpolation, expand, center, **opts)


@torch.jit.unused
def F_pil_perspective(
    img: Image.Image,
    perspective_coeffs: List[float],
    interpolation: int = BICUBIC,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    opts = F_pil__parse_fill(fill, img)

    return img.transform(img.size, PERSPECTIVE, perspective_coeffs, interpolation, **opts)


@torch.jit.unused
def F_pil_to_grayscale(img: Image.Image, num_output_channels: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if num_output_channels == 1:
        img = img.convert("L")
    elif num_output_channels == 3:
        img = img.convert("L")
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, "RGB")
    else:
        raise ValueError("num_output_channels should be either 1 or 3")

    return img


@torch.jit.unused
def F_pil_invert(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    return ImageOps.invert(img)


@torch.jit.unused
def F_pil_posterize(img: Image.Image, bits: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    return ImageOps.posterize(img, bits)


@torch.jit.unused
def F_pil_solarize(img: Image.Image, threshold: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    return ImageOps.solarize(img, threshold)


@torch.jit.unused
def F_pil_adjust_sharpness(img: Image.Image, sharpness_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    return img


@torch.jit.unused
def F_pil_autocontrast(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    return ImageOps.autocontrast(img)


@torch.jit.unused
def F_pil_equalize(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    return ImageOps.equalize(img)


import math
import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``,
    and ``lanczos``.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


# TODO: Once torchscript supports Enums with staticmethod
# this can be put into InterpolationMode as staticmethod
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.NEAREST_EXACT: 0,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

_is_pil_image = F_pil__is_pil_image


def get_dimensions(img: Tensor) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image dimensions.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(get_dimensions)
    if isinstance(img, torch.Tensor):
        return F_t_get_dimensions(img)

    return F_pil_get_dimensions(img)


def get_image_size(img: Tensor) -> List[int]:
    """Returns the size of an image as [width, height].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image size.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(get_image_size)
    if isinstance(img, torch.Tensor):
        return F_t_get_image_size(img)

    return F_pil_get_image_size(img)


def get_image_num_channels(img: Tensor) -> int:
    """Returns the number of channels of an image.

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        int: The number of channels.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(get_image_num_channels)
    if isinstance(img, torch.Tensor):
        return F_t_get_image_num_channels(img)

    return F_pil_get_image_num_channels(img)


@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic) -> Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(to_tensor)
    if not (F_pil__is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], F_pil_get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img


def pil_to_tensor(pic: Any) -> Tensor:
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    .. note::

        A deep copy of the underlying array is performed.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(pil_to_tensor)
    if not F_pil__is_pil_image(pic):
        raise TypeError(f"pic should be PIL Image. Got {type(pic)}")

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.array(pic, copy=True))
    img = img.view(pic.size[1], pic.size[0], F_pil_get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly
    This function does not support PIL Image.

    Args:
        image (torch.Tensor): Image to be converted
        dtype (torch.dtype): Desired data type of the output

    Returns:
        Tensor: Converted image

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(convert_image_dtype)
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input img should be Tensor Image")

    return F_t_convert_image_dtype(image, dtype)


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(to_pil_image)

    if not (isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndimension()} dimensions.")

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

        # check number of channels
        if pic.shape[-3] > 4:
            raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-3]} channels.")

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

        # check number of channels
        if pic.shape[-1] > 4:
            raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != "F":
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError("Input pic must be a torch.Tensor or NumPy ndarray, not {type(npimg)}")

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(normalize)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")

    return F_t_normalize(tensor, mean=mean, std=std, inplace=inplace)


def _compute_resized_output_size(
    image_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> List[int]:
    if len(size) == 1:  # specified size only for the smallest edge
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]


def resize(
    img: Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = "warn",
) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image. If the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``,
            ``size`` will be overruled so that the longer edge is equal to
            ``max_size``.
            As a result, the smaller edge may be shorter than ``size``. This
            is only supported if ``size`` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(resize)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    _, image_height, image_width = get_dimensions(img)
    if isinstance(size, int):
        size = [size]
    output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

    if [image_height, image_width] == output_size:
        return img

    antialias = _check_antialias(img, antialias, interpolation)

    if not isinstance(img, torch.Tensor):
        if antialias is False:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil_resize(img, size=output_size, interpolation=pil_interpolation)

    return F_t_resize(img, size=output_size, interpolation=interpolation.value, antialias=antialias)


def pad(img: Tensor, padding: List[int], fill: Union[int, float] = 0, padding_mode: str = "constant") -> Tensor:
    r"""Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor: Padded image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(pad)
    if not isinstance(img, torch.Tensor):
        return F_pil_pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    return F_t_pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(crop)
    if not isinstance(img, torch.Tensor):
        return F_pil_crop(img, top, left, height, width)

    return F_t_crop(img, top, left, height, width)


def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(center_crop)
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        _, image_height, image_width = get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
) -> Tensor:
    """Crop the given image and resize it to desired size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(resized_crop)
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation, antialias=antialias)
    return img


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Horizontally flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(hflip)
    if not isinstance(img, torch.Tensor):
        return F_pil_hflip(img)

    return F_t_hflip(img)


def _get_perspective_coeffs(startpoints: List[List[int]], endpoints: List[List[int]]) -> List[float]:
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution

    output: List[float] = res.tolist()
    return output


def perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Perform perspective transform of the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor: transformed Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(perspective)

    coeffs = _get_perspective_coeffs(startpoints, endpoints)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil_perspective(img, coeffs, interpolation=pil_interpolation, fill=fill)

    return F_t_perspective(img, coeffs, interpolation=interpolation.value, fill=fill)


def vflip(img: Tensor) -> Tensor:
    """Vertically flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Vertically flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(vflip)
    if not isinstance(img, torch.Tensor):
        return F_pil_vflip(img)

    return F_t_vflip(img)


def five_crop(img: Tensor, size: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
       Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(five_crop)
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    _, image_height, image_width = get_dimensions(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_height, crop_width)
    tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
    br = crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)

    center = center_crop(img, [crop_height, crop_width])

    return tl, tr, bl, br, center


def ten_crop(
    img: Tensor, size: List[int], vertical_flip: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate ten cropped images from the given image.
    Crop the given image into four corners and the central crop plus the
    flipped version of these (horizontal flipping is used by default).
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
        tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
        Corresponding top left, top right, bottom left, bottom right and
        center crop and same for the flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(ten_crop)
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    """Adjust brightness of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non-negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image or Tensor: Brightness adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(adjust_brightness)
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_brightness(img, brightness_factor)

    return F_t_adjust_brightness(img, brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image or Tensor: Contrast adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass#(adjust_contrast)
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_contrast(img, contrast_factor)

    return F_t_adjust_contrast(img, contrast_factor)


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    """Adjust color saturation of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image or Tensor: Saturation adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_saturation(img, saturation_factor)

    return F_t_adjust_saturation(img, saturation_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.
            Note: the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image or Tensor: Hue adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_hue(img, hue_factor)

    return F_t_adjust_hue(img, hue_factor)


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    r"""Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (PIL Image or Tensor): PIL Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, modes with transparency (alpha channel) are not supported.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    Returns:
        PIL Image or Tensor: Gamma correction adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_gamma(img, gamma, gain)

    return F_t_adjust_gamma(img, gamma, gain)


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def rotate(
    img: Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[int]] = None,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (number): rotation angle value in degrees, counter-clockwise.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
    Returns:
        PIL Image or Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil_rotate(img, angle=angle, interpolation=pil_interpolation, expand=expand, center=center, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return F_t_rotate(img, matrix=matrix, interpolation=interpolation.value, expand=expand, fill=fill)


def affine(
    img: Tensor,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x-axis, while
            the second value corresponds to a shear parallel to the y-axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        PIL Image or Tensor: Transformed image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    _, height, width = get_dimensions(img)
    if not isinstance(img, torch.Tensor):
        # center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t_affine
        if center is None:
            center = [width * 0.5, height * 0.5]
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil_affine(img, matrix=matrix, interpolation=pil_interpolation, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    return F_t_affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill)


# Looks like to_grayscale() is a stand-alone functional that is never called
# from the transform classes. Perhaps it's still here for BC? I can't be
# bothered to dig. Anyway, this can be deprecated as we migrate to V2.
@torch.jit.unused
def to_grayscale(img, num_output_channels=1):
    """Convert PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.
    This transform does not support torch Tensor.

    Args:
        img (PIL Image): PIL Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default is 1.

    Returns:
        PIL Image: Grayscale version of the image.

        - if num_output_channels = 1 : returned image is single channel
        - if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if isinstance(img, Image.Image):
        return F_pil_to_grayscale(img, num_output_channels)

    raise TypeError("Input should be PIL Image")


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    """Convert RGB image to grayscale version of image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Note:
        Please, note that this method supports only RGB images as input. For inputs in other color spaces,
        please, consider using meth:`~torchvision.transforms.functional.to_grayscale` with PIL Image.

    Args:
        img (PIL Image or Tensor): RGB Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        PIL Image or Tensor: Grayscale version of the image.

        - if num_output_channels = 1 : returned image is single channel
        - if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_to_grayscale(img, num_output_channels)

    return F_t_rgb_to_grayscale(img, num_output_channels)


def erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
    """Erase the input Tensor Image with given value.
    This transform does not support PIL Image.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default, is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img should be Tensor Image. Got {type(img)}")

    return F_t_erase(img, i, j, h, w, v, inplace=inplace)


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the image by given kernel.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.

            .. note::
                In torchscript mode kernel_size as single int is not supported, use a sequence of
                length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None.

            .. note::
                In torchscript mode sigma as single float is
                not supported, use a sequence of length 1: ``[sigma, ]``.

    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}")
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not F_pil__is_pil_image(img):
            raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")

        t_img = pil_to_tensor(img)

    output = F_t_gaussian_blur(t_img, kernel_size, sigma)

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output, mode=img.mode)
    return output


def invert(img: Tensor) -> Tensor:
    """Invert the colors of an RGB/grayscale image.

    Args:
        img (PIL Image or Tensor): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor: Color inverted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_invert(img)

    return F_t_invert(img)


def posterize(img: Tensor, bits: int) -> Tensor:
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        img (PIL Image or Tensor): Image to have its colors posterized.
            If img is torch Tensor, it should be of type torch.uint8, and
            it is expected to be in [..., 1 or 3, H, W] format, where ... means
            it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        bits (int): The number of bits to keep for each channel (0-8).
    Returns:
        PIL Image or Tensor: Posterized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not (0 <= bits <= 8):
        raise ValueError(f"The number if bits should be between 0 and 8. Got {bits}")

    if not isinstance(img, torch.Tensor):
        return F_pil_posterize(img, bits)

    return F_t_posterize(img, bits)


def solarize(img: Tensor, threshold: float) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        img (PIL Image or Tensor): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        threshold (float): All pixels equal or above this value are inverted.
    Returns:
        PIL Image or Tensor: Solarized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_solarize(img, threshold)

    return F_t_solarize(img, threshold)


def adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    """Adjust the sharpness of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.

    Returns:
        PIL Image or Tensor: Sharpness adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_adjust_sharpness(img, sharpness_factor)

    return F_t_adjust_sharpness(img, sharpness_factor)


def autocontrast(img: Tensor) -> Tensor:
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        img (PIL Image or Tensor): Image on which autocontrast is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor: An image that was autocontrasted.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_autocontrast(img)

    return F_t_autocontrast(img)


def equalize(img: Tensor) -> Tensor:
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        img (PIL Image or Tensor): Image on which equalize is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            The tensor dtype must be ``torch.uint8`` and values are expected to be in ``[0, 255]``.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Returns:
        PIL Image or Tensor: An image that was equalized.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    if not isinstance(img, torch.Tensor):
        return F_pil_equalize(img)

    return F_t_equalize(img)


def elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        img (PIL Image or Tensor): Image on which elastic_transform is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".
        displacement (Tensor): The displacement field. Expected shape is [1, H, W, 2].
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        pass
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not F_pil__is_pil_image(img):
            raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
        t_img = pil_to_tensor(img)

    shape = t_img.shape
    shape = (1,) + shape[-2:] + (2,)
    if shape != displacement.shape:
        raise ValueError(f"Argument displacement shape should be {shape}, but given {displacement.shape}")

    # TODO: if image shape is [N1, N2, ..., C, H, W] and
    # displacement is [1, H, W, 2] we need to reshape input image
    # such grid_sampler takes internal code for 4D input

    output = F_t_elastic_transform(
        t_img,
        displacement,
        interpolation=interpolation.value,
        fill=fill,
    )

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output, mode=img.mode)
    return output


# TODO in v0.17: remove this helper and change default of antialias to True everywhere
def _check_antialias(
    img: Tensor, antialias: Optional[Union[str, bool]], interpolation: InterpolationMode
) -> Optional[bool]:
    if isinstance(antialias, str):  # it should be "warn", but we don't bother checking against that
        if isinstance(img, Tensor) and (
            interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC
        ):
            warnings.warn(
                "The default value of the antialias parameter of all the resizing transforms "
                "(Resize(), RandomResizedCrop(), etc.) "
                "will change from None to True in v0.17, "
                "in order to be consistent across the PIL and Tensor backends. "
                "To suppress this warning, directly pass "
                "antialias=True (recommended, future default), antialias=None (current default, "
                "which means False for Tensors and True for PIL), "
                "or antialias=False (only works on Tensors - PIL will still use antialiasing). "
                "This also applies if you are using the inference transforms from the models weights: "
                "update the call to weights.transforms(antialias=True)."
            )
        antialias = None

    return antialias



from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from torch import functional as F


__all__ = [
    "ObjectDetection",
    "ImageClassification",
    "VideoClassification",
    "SemanticSegmentation",
    "OpticalFlow",
]


class ObjectDetection(nn.Module):
    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, torch.float)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[0.0, 1.0]``."
        )


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


class VideoClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: Tuple[int, int],
        resize_size: Tuple[int, int],
        mean: Tuple[float, ...] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, ...] = (0.22803, 0.22145, 0.216989),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        self.resize_size = list(resize_size)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, vid: Tensor) -> Tensor:
        need_squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            need_squeeze = True

        N, T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)
        vid = F.resize(vid, self.resize_size, interpolation=self.interpolation)
        vid = F.center_crop(vid, self.crop_size)
        vid = F.convert_image_dtype(vid, torch.float)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        H, W = self.crop_size
        vid = vid.view(N, T, C, H, W)
        vid = vid.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) => (N, C, T, H, W)

        if need_squeeze:
            vid = vid.squeeze(dim=0)
        return vid

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts batched ``(B, T, C, H, W)`` and single ``(T, C, H, W)`` video frame ``torch.Tensor`` objects. "
            f"The frames are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``. Finally the output "
            "dimensions are permuted to ``(..., C, T, H, W)`` tensors."
        )


class SemanticSegmentation(nn.Module):
    def __init__(
        self,
        *,
        resize_size: Optional[int],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )


class OpticalFlow(nn.Module):
    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )
    

    from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torch import functional as F


__all__ = [
    "ObjectDetection",
    "ImageClassification",
    "VideoClassification",
    "SemanticSegmentation",
    "OpticalFlow",
]


class ObjectDetection(nn.Module):
    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, torch.float)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[0.0, 1.0]``."
        )


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


class VideoClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: Tuple[int, int],
        resize_size: Tuple[int, int],
        mean: Tuple[float, ...] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, ...] = (0.22803, 0.22145, 0.216989),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        self.resize_size = list(resize_size)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, vid: Tensor) -> Tensor:
        need_squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            need_squeeze = True

        N, T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)
        # We hard-code antialias=False to preserve results after we changed
        # its default from None to True (see
        # https://github.com/pytorch/vision/pull/7160)
        # TODO: we could re-train the video models with antialias=True?
        vid = F.resize(vid, self.resize_size, interpolation=self.interpolation, antialias=False)
        vid = F.center_crop(vid, self.crop_size)
        vid = F.convert_image_dtype(vid, torch.float)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        H, W = self.crop_size
        vid = vid.view(N, T, C, H, W)
        vid = vid.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) => (N, C, T, H, W)

        if need_squeeze:
            vid = vid.squeeze(dim=0)
        return vid

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts batched ``(B, T, C, H, W)`` and single ``(T, C, H, W)`` video frame ``torch.Tensor`` objects. "
            f"The frames are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``. Finally the output "
            "dimensions are permuted to ``(..., C, T, H, W)`` tensors."
        )


class SemanticSegmentation(nn.Module):
    def __init__(
        self,
        *,
        resize_size: Optional[int],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )


class OpticalFlow(nn.Module):
    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )
    


import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from itertools import repeat
import collections


interpolate = torch.nn.functional.interpolate

def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"



class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )



class Conv3dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution3d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input video.
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm3d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm3d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv3d,
        )



class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input



class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)



class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)
    


import importlib
import inspect
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, TypeVar, Union

from torch import nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ["WeightsEnum", "Weights", "get_model", "get_model_builder", "get_model_weights", "get_weight", "list_models"]


@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    transforms: Callable
    meta: Dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        # We need this custom implementation for correct deep-copy and deserialization behavior.
        # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
        # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
        # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
        # for it, the check against the defined members would fail and effectively prevent the weights from being
        # deep-copied or deserialized.
        # See https://github.com/pytorch/vision/pull/7107 for details.
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False

        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta


def get_weight(name: str) -> WeightsEnum:
    """
    Gets the weights enum value by its full name. Example: "ResNet50_Weights.IMAGENET1K_V1"

    Args:
        name (str): The name of the weight enum entry.

    Returns:
        WeightsEnum: The requested weight enum.
    """
    try:
        enum_name, value_name = name.split(".")
    except ValueError:
        raise ValueError(f"Invalid weight name provided: '{name}'.")

    base_module_name = ".".join(sys.modules[__name__].__name__.split(".")[:-1])
    base_module = importlib.import_module(base_module_name)
    model_modules = [base_module] + [
        x[1]
        for x in inspect.getmembers(base_module, inspect.ismodule)
        if x[1].__file__.endswith("__init__.py")  # type: ignore[union-attr]
    ]

    weights_enum = None
    for m in model_modules:
        potential_class = m.__dict__.get(enum_name, None)
        if potential_class is not None and issubclass(potential_class, WeightsEnum):
            weights_enum = potential_class
            break

    if weights_enum is None:
        raise ValueError(f"The weight enum '{enum_name}' for the specific method couldn't be retrieved.")

    return weights_enum[value_name]


def get_model_weights(name: Union[Callable, str]) -> Type[WeightsEnum]:
    """
    Returns the weights enum class associated to the given model.

    Args:
        name (callable or str): The model builder function or the name under which it is registered.

    Returns:
        weights_enum (WeightsEnum): The weights enum class associated with the model.
    """
    model = get_model_builder(name) if isinstance(name, str) else name
    return _get_enum_from_fn(model)


def _get_enum_from_fn(fn: Callable) -> Type[WeightsEnum]:
    """
    Internal method that gets the weight enum of a specific model builder method.

    Args:
        fn (Callable): The builder method used to create the model.
    Returns:
        WeightsEnum: The requested weight enum.
    """
    sig = signature(fn)
    if "weights" not in sig.parameters:
        raise ValueError("The method is missing the 'weights' argument.")

    ann = signature(fn).parameters["weights"].annotation
    weights_enum = None
    if isinstance(ann, type) and issubclass(ann, WeightsEnum):
        weights_enum = ann
    else:
        # handle cases like Union[Optional, T]
        # TODO: Replace ann.__args__ with typing.get_args(ann) after python >= 3.8
        for t in ann.__args__:  # type: ignore[union-attr]
            if isinstance(t, type) and issubclass(t, WeightsEnum):
                weights_enum = t
                break

    if weights_enum is None:
        raise ValueError(
            "The WeightsEnum class for the specific method couldn't be retrieved. Make sure the typing info is correct."
        )

    return weights_enum


M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}


def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


def list_models(module: Optional[ModuleType] = None) -> List[str]:
    """
    Returns a list with the names of registered models.

    Args:
        module (ModuleType, optional): The module from which we want to extract the available models.

    Returns:
        models (list): A list with the names of available models.
    """
    models = [
        k for k, v in BUILTIN_MODELS.items() if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    ]
    return sorted(models)


def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def get_model(name: str, **config: Any) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    Args:
        name (str): The name under which the model is registered.
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_model_builder(name)
    return fn(**config)



IMAGENET_CATEGORIES = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead",
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "European fire salamander",
    "common newt",
    "eft",
    "spotted salamander",
    "axolotl",
    "bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead",
    "leatherback turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "common iguana",
    "American chameleon",
    "whiptail",
    "agama",
    "frilled lizard",
    "alligator lizard",
    "Gila monster",
    "green lizard",
    "African chameleon",
    "Komodo dragon",
    "African crocodile",
    "American alligator",
    "triceratops",
    "thunder snake",
    "ringneck snake",
    "hognose snake",
    "green snake",
    "king snake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "horned viper",
    "diamondback",
    "sidewinder",
    "trilobite",
    "harvestman",
    "scorpion",
    "black and gold garden spider",
    "barn spider",
    "garden spider",
    "black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie chicken",
    "peacock",
    "quail",
    "partridge",
    "African grey",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "Dungeness crab",
    "rock crab",
    "fiddler crab",
    "king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "American egret",
    "bittern",
    "crane bird",
    "limpkin",
    "European gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "red-backed sandpiper",
    "redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound",
    "English foxhound",
    "redbone",
    "borzoi",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound",
    "Norwegian elkhound",
    "otterhound",
    "Saluki",
    "Scottish deerhound",
    "Weimaraner",
    "Staffordshire bullterrier",
    "American Staffordshire terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier",
    "Airedale",
    "cairn",
    "Australian terrier",
    "Dandie Dinmont",
    "Boston bull",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier",
    "Tibetan terrier",
    "silky terrier",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla",
    "English setter",
    "Irish setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber",
    "English springer",
    "Welsh springer spaniel",
    "cocker spaniel",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog",
    "Shetland sheepdog",
    "collie",
    "Border collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German shepherd",
    "Doberman",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard",
    "Eskimo dog",
    "malamute",
    "Siberian husky",
    "dalmatian",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke",
    "Cardigan",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf",
    "white wolf",
    "red wolf",
    "coyote",
    "dingo",
    "dhole",
    "African hunting dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian cat",
    "cougar",
    "lynx",
    "leopard",
    "snow leopard",
    "jaguar",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "American black bear",
    "ice bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "long-horned beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket",
    "walking stick",
    "cockroach",
    "mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "admiral",
    "ringlet",
    "monarch",
    "cabbage butterfly",
    "sulphur butterfly",
    "lycaenid",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "wood rabbit",
    "hare",
    "Angora",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "sorrel",
    "zebra",
    "hog",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram",
    "bighorn",
    "ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "Arabian camel",
    "llama",
    "weasel",
    "mink",
    "polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas",
    "baboon",
    "macaque",
    "langur",
    "colobus",
    "proboscis monkey",
    "marmoset",
    "capuchin",
    "howler monkey",
    "titi",
    "spider monkey",
    "squirrel monkey",
    "Madagascar cat",
    "indri",
    "Indian elephant",
    "African elephant",
    "lesser panda",
    "giant panda",
    "barracouta",
    "eel",
    "coho",
    "rock beauty",
    "anemone fish",
    "sturgeon",
    "gar",
    "lionfish",
    "puffer",
    "abacus",
    "abaya",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibian",
    "analog clock",
    "apiary",
    "apron",
    "ashcan",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint",
    "Band Aid",
    "banjo",
    "bannister",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "barrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "bathing cap",
    "bath towel",
    "bathtub",
    "beach wagon",
    "beacon",
    "beaker",
    "bearskin",
    "beer bottle",
    "beer glass",
    "bell cote",
    "bib",
    "bicycle-built-for-two",
    "bikini",
    "binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsled",
    "bolo tie",
    "bonnet",
    "bookcase",
    "bookshop",
    "bottlecap",
    "bow",
    "bow tie",
    "brass",
    "brassiere",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "bullet train",
    "butcher shop",
    "cab",
    "caldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "carpenter's kit",
    "carton",
    "car wheel",
    "cash machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "cellular telephone",
    "chain",
    "chainlink fence",
    "chain mail",
    "chain saw",
    "chest",
    "chiffonier",
    "chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "cinema",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clog",
    "cocktail shaker",
    "coffee mug",
    "coffeepot",
    "coil",
    "combination lock",
    "computer keyboard",
    "confectionery",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "crane",
    "crash helmet",
    "crate",
    "crib",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishrag",
    "dishwasher",
    "disk brake",
    "dock",
    "dogsled",
    "dome",
    "doormat",
    "drilling platform",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso maker",
    "face powder",
    "feather boa",
    "file",
    "fireboat",
    "fire engine",
    "fire screen",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "gasmask",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golfcart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "grille",
    "grocery store",
    "guillotine",
    "hair slide",
    "hair spray",
    "half track",
    "hammer",
    "hamper",
    "hand blower",
    "hand-held computer",
    "handkerchief",
    "hard disc",
    "harmonica",
    "harp",
    "harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoopskirt",
    "horizontal bar",
    "horse cart",
    "hourglass",
    "iPod",
    "iron",
    "jack-o'-lantern",
    "jean",
    "jeep",
    "jersey",
    "jigsaw puzzle",
    "jinrikisha",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop",
    "lawn mower",
    "lens cap",
    "letter opener",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "liner",
    "lipstick",
    "Loafer",
    "lotion",
    "loudspeaker",
    "loupe",
    "lumbermill",
    "magnetic compass",
    "mailbag",
    "mailbox",
    "maillot",
    "maillot tank suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "matchstick",
    "maypole",
    "maze",
    "measuring cup",
    "medicine chest",
    "megalith",
    "microphone",
    "microwave",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "mortarboard",
    "mosque",
    "mosquito net",
    "motor scooter",
    "mountain bike",
    "mountain tent",
    "mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "organ",
    "oscilloscope",
    "overskirt",
    "oxcart",
    "oxygen mask",
    "packet",
    "paddle",
    "paddlewheel",
    "padlock",
    "paintbrush",
    "pajama",
    "palace",
    "panpipe",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "passenger car",
    "patio",
    "pay-phone",
    "pedestal",
    "pencil box",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "pick",
    "pickelhaube",
    "picket fence",
    "pickup",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate",
    "pitcher",
    "plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow",
    "plunger",
    "Polaroid camera",
    "pole",
    "police van",
    "poncho",
    "pool table",
    "pop bottle",
    "pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "projectile",
    "projector",
    "puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "racer",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "rubber eraser",
    "rugby ball",
    "rule",
    "running shoe",
    "safe",
    "safety pin",
    "saltshaker",
    "sandal",
    "sarong",
    "sax",
    "scabbard",
    "scale",
    "school bus",
    "schooner",
    "scoreboard",
    "screen",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe shop",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar dish",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "speedboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "steel arch bridge",
    "steel drum",
    "stethoscope",
    "stole",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "streetcar",
    "stretcher",
    "studio couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglass",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "swab",
    "sweatshirt",
    "swimming trunks",
    "swing",
    "switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy",
    "television",
    "tennis ball",
    "thatch",
    "theater curtain",
    "thimble",
    "thresher",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toyshop",
    "tractor",
    "trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright",
    "vacuum",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "warplane",
    "washbasin",
    "washer",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool",
    "worm fence",
    "wreck",
    "yawl",
    "yurt",
    "web site",
    "comic book",
    "crossword puzzle",
    "street sign",
    "traffic light",
    "book jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "ice lolly",
    "French loaf",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hotdog",
    "mashed potato",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate sauce",
    "dough",
    "meat loaf",
    "pizza",
    "potpie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeside",
    "promontory",
    "sandbar",
    "seashore",
    "valley",
    "volcano",
    "ballplayer",
    "groom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "hip",
    "buckeye",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn",
    "earthstar",
    "hen-of-the-woods",
    "bolete",
    "ear",
    "toilet tissue",
]

# To be replaced with torchvision.datasets.info("coco").categories
_COCO_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# To be replaced with torchvision.datasets.info("coco_kp")
_COCO_PERSON_CATEGORIES = ["no person", "person"]
_COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# To be replaced with torchvision.datasets.info("voc").categories
_VOC_CATEGORIES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# To be replaced with torchvision.datasets.info("kinetics400").categories
_KINETICS400_CATEGORIES = [
    "abseiling",
    "air drumming",
    "answering questions",
    "applauding",
    "applying cream",
    "archery",
    "arm wrestling",
    "arranging flowers",
    "assembling computer",
    "auctioning",
    "baby waking up",
    "baking cookies",
    "balloon blowing",
    "bandaging",
    "barbequing",
    "bartending",
    "beatboxing",
    "bee keeping",
    "belly dancing",
    "bench pressing",
    "bending back",
    "bending metal",
    "biking through snow",
    "blasting sand",
    "blowing glass",
    "blowing leaves",
    "blowing nose",
    "blowing out candles",
    "bobsledding",
    "bookbinding",
    "bouncing on trampoline",
    "bowling",
    "braiding hair",
    "breading or breadcrumbing",
    "breakdancing",
    "brush painting",
    "brushing hair",
    "brushing teeth",
    "building cabinet",
    "building shed",
    "bungee jumping",
    "busking",
    "canoeing or kayaking",
    "capoeira",
    "carrying baby",
    "cartwheeling",
    "carving pumpkin",
    "catching fish",
    "catching or throwing baseball",
    "catching or throwing frisbee",
    "catching or throwing softball",
    "celebrating",
    "changing oil",
    "changing wheel",
    "checking tires",
    "cheerleading",
    "chopping wood",
    "clapping",
    "clay pottery making",
    "clean and jerk",
    "cleaning floor",
    "cleaning gutters",
    "cleaning pool",
    "cleaning shoes",
    "cleaning toilet",
    "cleaning windows",
    "climbing a rope",
    "climbing ladder",
    "climbing tree",
    "contact juggling",
    "cooking chicken",
    "cooking egg",
    "cooking on campfire",
    "cooking sausages",
    "counting money",
    "country line dancing",
    "cracking neck",
    "crawling baby",
    "crossing river",
    "crying",
    "curling hair",
    "cutting nails",
    "cutting pineapple",
    "cutting watermelon",
    "dancing ballet",
    "dancing charleston",
    "dancing gangnam style",
    "dancing macarena",
    "deadlifting",
    "decorating the christmas tree",
    "digging",
    "dining",
    "disc golfing",
    "diving cliff",
    "dodgeball",
    "doing aerobics",
    "doing laundry",
    "doing nails",
    "drawing",
    "dribbling basketball",
    "drinking",
    "drinking beer",
    "drinking shots",
    "driving car",
    "driving tractor",
    "drop kicking",
    "drumming fingers",
    "dunking basketball",
    "dying hair",
    "eating burger",
    "eating cake",
    "eating carrots",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "eating spaghetti",
    "eating watermelon",
    "egg hunting",
    "exercising arm",
    "exercising with an exercise ball",
    "extinguishing fire",
    "faceplanting",
    "feeding birds",
    "feeding fish",
    "feeding goats",
    "filling eyebrows",
    "finger snapping",
    "fixing hair",
    "flipping pancake",
    "flying kite",
    "folding clothes",
    "folding napkins",
    "folding paper",
    "front raises",
    "frying vegetables",
    "garbage collecting",
    "gargling",
    "getting a haircut",
    "getting a tattoo",
    "giving or receiving award",
    "golf chipping",
    "golf driving",
    "golf putting",
    "grinding meat",
    "grooming dog",
    "grooming horse",
    "gymnastics tumbling",
    "hammer throw",
    "headbanging",
    "headbutting",
    "high jump",
    "high kick",
    "hitting baseball",
    "hockey stop",
    "holding snake",
    "hopscotch",
    "hoverboarding",
    "hugging",
    "hula hooping",
    "hurdling",
    "hurling (sport)",
    "ice climbing",
    "ice fishing",
    "ice skating",
    "ironing",
    "javelin throw",
    "jetskiing",
    "jogging",
    "juggling balls",
    "juggling fire",
    "juggling soccer ball",
    "jumping into pool",
    "jumpstyle dancing",
    "kicking field goal",
    "kicking soccer ball",
    "kissing",
    "kitesurfing",
    "knitting",
    "krumping",
    "laughing",
    "laying bricks",
    "long jump",
    "lunge",
    "making a cake",
    "making a sandwich",
    "making bed",
    "making jewelry",
    "making pizza",
    "making snowman",
    "making sushi",
    "making tea",
    "marching",
    "massaging back",
    "massaging feet",
    "massaging legs",
    "massaging person's head",
    "milking cow",
    "mopping floor",
    "motorcycling",
    "moving furniture",
    "mowing lawn",
    "news anchoring",
    "opening bottle",
    "opening present",
    "paragliding",
    "parasailing",
    "parkour",
    "passing American football (in game)",
    "passing American football (not in game)",
    "peeling apples",
    "peeling potatoes",
    "petting animal (not cat)",
    "petting cat",
    "picking fruit",
    "planting trees",
    "plastering",
    "playing accordion",
    "playing badminton",
    "playing bagpipes",
    "playing basketball",
    "playing bass guitar",
    "playing cards",
    "playing cello",
    "playing chess",
    "playing clarinet",
    "playing controller",
    "playing cricket",
    "playing cymbals",
    "playing didgeridoo",
    "playing drums",
    "playing flute",
    "playing guitar",
    "playing harmonica",
    "playing harp",
    "playing ice hockey",
    "playing keyboard",
    "playing kickball",
    "playing monopoly",
    "playing organ",
    "playing paintball",
    "playing piano",
    "playing poker",
    "playing recorder",
    "playing saxophone",
    "playing squash or racquetball",
    "playing tennis",
    "playing trombone",
    "playing trumpet",
    "playing ukulele",
    "playing violin",
    "playing volleyball",
    "playing xylophone",
    "pole vault",
    "presenting weather forecast",
    "pull ups",
    "pumping fist",
    "pumping gas",
    "punching bag",
    "punching person (boxing)",
    "push up",
    "pushing car",
    "pushing cart",
    "pushing wheelchair",
    "reading book",
    "reading newspaper",
    "recording music",
    "riding a bike",
    "riding camel",
    "riding elephant",
    "riding mechanical bull",
    "riding mountain bike",
    "riding mule",
    "riding or walking with horse",
    "riding scooter",
    "riding unicycle",
    "ripping paper",
    "robot dancing",
    "rock climbing",
    "rock scissors paper",
    "roller skating",
    "running on treadmill",
    "sailing",
    "salsa dancing",
    "sanding floor",
    "scrambling eggs",
    "scuba diving",
    "setting table",
    "shaking hands",
    "shaking head",
    "sharpening knives",
    "sharpening pencil",
    "shaving head",
    "shaving legs",
    "shearing sheep",
    "shining shoes",
    "shooting basketball",
    "shooting goal (soccer)",
    "shot put",
    "shoveling snow",
    "shredding paper",
    "shuffling cards",
    "side kick",
    "sign language interpreting",
    "singing",
    "situp",
    "skateboarding",
    "ski jumping",
    "skiing (not slalom or crosscountry)",
    "skiing crosscountry",
    "skiing slalom",
    "skipping rope",
    "skydiving",
    "slacklining",
    "slapping",
    "sled dog racing",
    "smoking",
    "smoking hookah",
    "snatch weight lifting",
    "sneezing",
    "sniffing",
    "snorkeling",
    "snowboarding",
    "snowkiting",
    "snowmobiling",
    "somersaulting",
    "spinning poi",
    "spray painting",
    "spraying",
    "springboard diving",
    "squat",
    "sticking tongue out",
    "stomping grapes",
    "stretching arm",
    "stretching leg",
    "strumming guitar",
    "surfing crowd",
    "surfing water",
    "sweeping floor",
    "swimming backstroke",
    "swimming breast stroke",
    "swimming butterfly stroke",
    "swing dancing",
    "swinging legs",
    "swinging on something",
    "sword fighting",
    "tai chi",
    "taking a shower",
    "tango dancing",
    "tap dancing",
    "tapping guitar",
    "tapping pen",
    "tasting beer",
    "tasting food",
    "testifying",
    "texting",
    "throwing axe",
    "throwing ball",
    "throwing discus",
    "tickling",
    "tobogganing",
    "tossing coin",
    "tossing salad",
    "training dog",
    "trapezing",
    "trimming or shaving beard",
    "trimming trees",
    "triple jump",
    "tying bow tie",
    "tying knot (not on a tie)",
    "tying tie",
    "unboxing",
    "unloading truck",
    "using computer",
    "using remote controller (not gaming)",
    "using segway",
    "vault",
    "waiting in line",
    "walking the dog",
    "washing dishes",
    "washing feet",
    "washing hair",
    "washing hands",
    "water skiing",
    "water sliding",
    "watering plants",
    "waxing back",
    "waxing chest",
    "waxing eyebrows",
    "waxing legs",
    "weaving basket",
    "welding",
    "whistling",
    "windsurfing",
    "wrapping present",
    "wrestling",
    "writing",
    "yawning",
    "yoga",
    "zumba",
]


import enum
from typing import Sequence, Type, TypeVar

T = TypeVar("T", bound=enum.Enum)


class StrEnumMeta(enum.EnumMeta):
    auto = enum.auto

    def from_str(self: Type[T], member: str) -> T:  # type: ignore[misc]
        try:
            return self[member]
        except KeyError:
            # TODO: use `add_suggestion` from torchvision.prototype.utils._internal to improve the error message as
            #  soon as it is migrated.
            raise ValueError(f"Unknown value '{member}' for {self.__name__}.") from None


class StrEnum(enum.Enum, metaclass=StrEnumMeta):
    pass


def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if not seq:
        return ""
    if len(seq) == 1:
        return f"'{seq[0]}'"

    head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
    tail = f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"

    return head + tail


import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


D = TypeVar("D")


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated since 0.13 and may be removed in the future. Please use keyword parameter(s) "
                f"instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper


W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with a dictionary of the keyword arguments. The only key that is guaranteed to be in
            the dictionary is the deprecated parameter name passed as first element in the tuple. All other parameters
            should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument already use
                # the new style arguments, there is nothing to do. Note that we cannot use `None` as sentinel for the
                # weight argument, since it is a valid value.
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw decorator uses the current
                # signature to infer the names of positionally passed arguments and thus has no knowledge that there
                # used to be a pretrained parameter.
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    # We put the pretrained argument under its legacy name in the keyword argument dictionary to have
                    # unified access to the value if the default value is a callable.
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, "
                        f"please use '{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated since 0.13 and "
                    f"may be removed in the future. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
    if actual is not None:
        if actual != expected:
            raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    return expected


class _ModelURLs(dict):
    def __getitem__(self, item):
        warnings.warn(
            "Accessing the model URLs via the internal dictionary of the module is deprecated since 0.13 and may "
            "be removed in the future. Please access them via the appropriate Weights Enum instead."
        )
        return super().__getitem__(item)
    

import torch
from torch import nn, Tensor


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if mode not in ["batch", "row"]:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    size = [1] * input.ndim
    if mode == "row":
        size[0] = input.shape[0]
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'p=' + str(self.p)
        tmpstr += ', mode=' + str(self.mode)
        tmpstr += ')'
        return tmpstr
    

import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

#from torchvision.ops.stochastic_depth import StochasticDepth
#from torchvision.models._api import register_model, Weights, WeightsEnum
#from torchvision.models._meta import _IMAGENET_CATEGORIES
#from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "SwinTransformer",
    "Swin_T_Weights",
    "Swin_S_Weights",
    "Swin_B_Weights",
    "Swin_V2_T_Weights",
    "Swin_V2_S_Weights",
    "Swin_V2_B_Weights",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]


def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x


#torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


#torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


#torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor):
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "categories": IMAGENET_CATEGORIES,
}


class Swin_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_t-704ceda3.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=232, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 28288354,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.474,
                    "acc@5": 95.776,
                }
            },
            "_ops": 4.491,
            "_file_size": 108.19,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_s-5e29d889.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=246, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 49606258,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.196,
                    "acc@5": 96.360,
                }
            },
            "_ops": 8.741,
            "_file_size": 189.786,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_b-68c6b09e.pth",
        transforms=partial(
            ImageClassification, crop_size=224, resize_size=238, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 87768224,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.582,
                    "acc@5": 96.640,
                }
            },
            "_ops": 15.431,
            "_file_size": 335.364,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 28351570,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.072,
                    "acc@5": 96.132,
                }
            },
            "_ops": 5.94,
            "_file_size": 108.626,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 49737442,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.712,
                    "acc@5": 96.816,
                }
            },
            "_ops": 11.546,
            "_file_size": 190.675,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class Swin_V2_B_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
        transforms=partial(
            ImageClassification, crop_size=256, resize_size=272, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META,
            "num_params": 87930848,
            "min_size": (256, 256),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.112,
                    "acc@5": 96.864,
                }
            },
            "_ops": 20.325,
            "_file_size": 336.372,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_T_Weights.IMAGENET1K_V1))
def swin_v2_t(*, weights: Optional[Swin_V2_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_T_Weights
        :members:
    """
    weights = Swin_V2_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_S_Weights.IMAGENET1K_V1))
def swin_v2_s(*, weights: Optional[Swin_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.!!
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_S_Weights
        :members:
    """
    weights = Swin_V2_S_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_B_Weights.IMAGENET1K_V1))
def swin_v2_b(*, weights: Optional[Swin_V2_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and Resolution <https://arxiv.org/abs/2111.09883>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_V2_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_V2_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_V2_B_Weights
        :members:
    """
    weights = Swin_V2_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )