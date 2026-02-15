import torch
import torch.nn as nn

class EDACLayer(nn.Module):
    """
    Error Detection and Correction (EDAC) layer that monitors
    activation ranges and applies correction using duplicated channels.
    """

    def __init__(self, num_channels, duplicated_idx=None):
        """
        Initializes the EDAC layer with per-channel min/max buffers
        and optional indices of duplicated channels.
        """
        super().__init__()
        self.duplicated_idx = duplicated_idx or []
        self.register_buffer("min_val",
                             torch.zeros(num_channels))
        self.register_buffer("max_val",
                             torch.zeros(num_channels))

    def profile(self, activation_tensor):
        """
        Profiles activations to compute and store per-channel
        minimum and maximum reference values.
        """
        self.min_val = activation_tensor.min(dim=0)[0]
        self.max_val = activation_tensor.max(dim=0)[0]

    def forward(self, x):
        """
        Performs runtime error detection and correction:
        - Replaces NaNs with zeros
        - Validates activations against profiled ranges
        - Uses duplicated channels for correction when available
        - Zeros out invalid activations
        """
        x = torch.nan_to_num(x, nan=0.0)

        for c in range(x.size(1)):

            if c in self.duplicated_idx:
                dup = c + len(self.duplicated_idx)

                v1 = x[:, c]
                v2 = x[:, dup]

                in1 = (v1 >= self.min_val[c]) & (v1 <= self.max_val[c])
                in2 = (v2 >= self.min_val[c]) & (v2 <= self.max_val[c])

                both = in1 & in2
                x[:, c][both] = torch.min(v1[both], v2[both])

                x[:, c][~in1 & in2] = v2[~in1 & in2]
                x[:, c][in1 & ~in2] = v1[in1 & ~in2]
                x[:, c][~in1 & ~in2] = 0.0

            else:
                out = (x[:, c] < self.min_val[c]) | \
                      (x[:, c] > self.max_val[c])
                x[:, c][out] = 0.0

        return x
