import torch
import contextlib

def infer_masks(
    model,
    batch,
    device,
    use_fp16: bool = False,
):
    """
    SAM2 mask decoder inference (embedding-level)
    """
    ctx = (
        torch.cuda.amp.autocast()
        if (use_fp16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with ctx:
        low_res_masks, _, _, _ = model(
            image_embeddings=batch[3].to(device, non_blocking=True),
            image_pe=batch[4].to(device, non_blocking=True),
            sparse_prompt_embeddings=batch[1].to(device, non_blocking=True),
            dense_prompt_embeddings=batch[2].to(device, non_blocking=True),
            multimask_output=True,
            repeat_image=False,
            high_res_features=[h.to(device, non_blocking=True) for h in batch[5]],
        )

    return low_res_masks


def postprocess_masks(low_res_masks: torch.Tensor) -> torch.Tensor:
    """
    Argmax to label mask (B, H, W)
    """
    return low_res_masks.argmax(dim=1)
