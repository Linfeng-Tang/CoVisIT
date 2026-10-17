import torch
import kornia.filters as KF

def Re_loss_t(
    img1,
    img2,
    t=None,
    num_timesteps=None,
    weights=(5.0, 5.0),
    power=2.0,
    intensity: str = "l1",
    grad: str = "l1",
):
    intensity = str(intensity).lower()
    grad = str(grad).lower()

    
    if intensity in ("mse", "l2"):
        int_map = (img1 - img2).square()
    else:
        int_map = torch.abs(img1 - img2)
    loss_int = int_map.mean()

    
    g1 = KF.spatial_gradient(img1, order=1).abs()
    g2 = KF.spatial_gradient(img2, order=1).abs()
    grad1 = g1.sum(dim=(1, 2))
    grad2 = g2.sum(dim=(1, 2))
    if grad in ("mse", "l2"):
        grad_map = (grad1 - grad2).square()
    else:
        grad_map = torch.abs(grad1 - grad2)
    loss_grad = grad_map.mean()

    loss_total = weights[0] * loss_int + weights[1] * loss_grad

    return loss_total, loss_int, loss_grad
