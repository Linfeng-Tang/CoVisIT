import numpy as np
import cv2
import torch

def _ssim(img1, img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _to_numpy_hwc(img):
    
    try:
        import torch
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.ndim == 4:  
                if x.shape[0] != 1:
                    raise ValueError(f"Only support batch=1, got {tuple(x.shape)}")
                x = x[0]
            if x.ndim == 3:  
                x = x.permute(1, 2, 0)  
            elif x.ndim == 2:
                x = x[:, :, None]       
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")
            img = x.numpy()
    except Exception:
        pass

    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type: {type(img)}")

    if img.ndim == 2:
        img = img[..., None]  
    if img.ndim != 3:
        raise ValueError(f"Expect HW or HWC, got {img.shape}")

    c = img.shape[2]
    if c not in (1, 3):
        raise ValueError(f"Expect C=1 or 3, got C={c}")

    return img.astype(np.float64)


def _normalize_to_255(img, input_range="auto"):
    if input_range == "auto":
        mx = float(np.max(img)) if img.size else 0.0
        input_range = "01" if mx <= 1.5 else "255"

    if input_range == "01":
        img = np.clip(img, 0.0, 1.0) * 255.0
    elif input_range == "255":
        img = np.clip(img, 0.0, 255.0)
    else:
        raise ValueError("input_range must be 'auto', '01', or '255'")
    return img


def _ensure_3ch(img_hwc):
    if img_hwc.shape[2] == 1:
        img_hwc = np.repeat(img_hwc, 3, axis=2)
    return img_hwc


def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False,
                   input_range="auto", color_order="rgb"):
    img1 = _to_numpy_hwc(img1)
    img2 = _to_numpy_hwc(img2)
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    img1 = _normalize_to_255(img1, input_range=input_range)
    img2 = _normalize_to_255(img2, input_range=input_range)

    
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    
    if test_y_channel:
        img1_3 = _ensure_3ch(img1)
        img2_3 = _ensure_3ch(img2)
        img1 = to_y_channel(img1_3, color_order=color_order)
        img2 = to_y_channel(img2_3, color_order=color_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False,
                   input_range="auto", color_order="rgb"):
    img1 = _to_numpy_hwc(img1)
    img2 = _to_numpy_hwc(img2)
    if img1.shape != img2.shape:
        raise ValueError(f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    img1 = _normalize_to_255(img1, input_range=input_range)
    img2 = _normalize_to_255(img2, input_range=input_range)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1_3 = _ensure_3ch(img1)
        img2_3 = _ensure_3ch(img2)
        img1 = to_y_channel(img1_3, color_order=color_order)
        img2 = to_y_channel(img2_3, color_order=color_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    
    if img1.shape[2] == 1:
        return float(_ssim(img1[..., 0], img2[..., 0]))

    
    ssims = [_ssim(img1[..., i], img2[..., i]) for i in range(3)]
    return float(np.array(ssims).mean())


def to_y_channel(img, color_order: str = "bgr"):
    img = img.astype(np.float64) / 255.

    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        return img * 255.

    if color_order.lower() == "bgr":
        y = bgr2ycbcr(img, y_only=True)
    elif color_order.lower() == "rgb":
        
        y = bgr2ycbcr(img[..., ::-1], y_only=True)
    else:
        raise ValueError("color_order must be 'bgr' or 'rgb'")

    return (y[..., None] * 255.)

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)