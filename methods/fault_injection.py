import torch

def inject_bitflips(model, ber):
    """
    Vectorized GPU-based bit-flip injection.
    Preserves the same fault model as the original implementation.
    """

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue

        flat = param.data.view(-1)

        total_bits = flat.numel() * 32
        n_flip = int(total_bits * ber)

        if n_flip == 0:
            continue

        # Random bit positions (on GPU)
        bit_positions = torch.randint(
            0, total_bits,
            (n_flip,),
            device=flat.device
        )

        param_idx = bit_positions // 32
        bit_offset = bit_positions % 32

        # View float32 as int32 WITHOUT copying
        int_view = flat.view(torch.int32)

        # Build bit masks
        masks = (1 << bit_offset).to(torch.int32)

        # Apply XOR (vectorized)
        int_view[param_idx] ^= masks
