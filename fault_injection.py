import torch
import numpy as np

def inject_bitflips(model, ber):
    """
    Injects random bit-flip faults into the model's weights
    based on the given Bit Error Rate (BER).
    """

    for name, param in model.named_parameters():

        if "weight" not in name:
            continue

        flat = param.data.view(-1)

        total_bits = flat.numel() * 32
        n_flip = int(total_bits * ber)

        if n_flip == 0:
            continue

        bit_positions = torch.randint(
            0, total_bits, (n_flip,),
            device=flat.device
        )

        param_idx = bit_positions // 32
        bit_offset = bit_positions % 32

        values = flat[param_idx].cpu().numpy().astype(np.float32)
        int_repr = values.view(np.uint32)

        for i in range(n_flip):
            int_repr[i] ^= (1 << int(bit_offset[i]))

        new_vals = int_repr.view(np.float32)

        flat[param_idx] = torch.tensor(
            new_vals,
            device=flat.device
        )
