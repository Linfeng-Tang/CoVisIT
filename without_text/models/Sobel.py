import cv2
import numpy as np
import os

def sobel_magnitude_condition(
    rgb_uint8: np.ndarray,
    ksize: int = 3,
    blur_ksize: int = 3,
    blur_sigma: float = 0.0,
    soft_threshold: float = 0.08,
    gamma: float = 0.8,
    save_path: str = None
) -> np.ndarray:
    assert rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3

    
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    
    mag = cv2.magnitude(gx, gy)

    
    mag = mag / (mag.max() + 1e-6)

    
    if soft_threshold is not None and soft_threshold > 0:
        mag = np.clip(
            (mag - soft_threshold) / (1.0 - soft_threshold),
            0.0,
            1.0
        )

    
    if gamma is not None and gamma > 0:
        mag = np.power(mag, gamma)

    
    
    mag_3ch = np.stack([mag, mag, mag], axis=0)

    
    if save_path is not None:
        mag_vis = (mag * 255.0).clip(0, 255).astype(np.uint8)
        mag_vis_3ch = np.stack([mag_vis]*3, axis=-1)  
        cv2.imwrite(save_path, mag_vis_3ch)

    
    return mag_3ch.astype(np.float32)



def sobel_magnitude_condition_from_path(
    img_path: str,
    ksize: int = 3,
    blur_ksize: int = 3,
    blur_sigma: float = 0.0,
    soft_threshold: float = 0.08,
    gamma: float = 0.8,
    save_path: str = None
) -> np.ndarray:

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    
    if rgb.shape[0] != 512 or rgb.shape[1] != 512:
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)

    
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    
    mag = cv2.magnitude(gx, gy)

    
    mag = mag / (mag.max() + 1e-6)

    
    if soft_threshold is not None and soft_threshold > 0:
        mag = np.clip(
            (mag - soft_threshold) / (1.0 - soft_threshold),
            0.0,
            1.0
        )

    
    if gamma is not None and gamma > 0:
        mag = np.power(mag, gamma)

    
    if save_path is not None:
        
        if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
            os.makedirs(save_path, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(save_path, base_name + "_sobel.png")

        
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_path = save_path

        mag_vis = (mag * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(out_path, mag_vis)

    
    return mag[None, ...].astype(np.float32)








































































































































































