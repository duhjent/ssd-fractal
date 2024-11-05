import numpy as np
import torch
from torch import nn

from .conv_block import ConvBlock


class FractalBlock(nn.Module):
    def __init__(
        self,
        n_columns,
        C_in,
        C_out,
        p_ldrop,
        p_dropout,
        pad_type="zero",
        doubling=False,
        dropout_pos="CDBR",
    ):
        """Fractal block
        Args:
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_ldrop: local droppath prob
            - p_dropout: dropout prob
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()

        self.n_columns = n_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        self.C_out = C_out
        if dropout_pos == "FD" and p_dropout > 0.0:
            self.dropout = nn.Dropout2d(p=p_dropout)
            p_dropout = 0.0
        else:
            self.dropout = None

        if doubling:
            self.doubler = ConvBlock(C_in, C_out, 1, padding=0)
        else:
            self.doubler = None

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns - 1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int32)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    first_block = i + 1 == dist  # first block in this column
                    if first_block and not doubling:
                        # if doubling, always input channel size is C_out.
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ConvBlock(
                        cur_C_in,
                        C_out,
                        dropout=p_dropout,
                        pad_type=pad_type,
                        dropout_pos=dropout_pos,
                    )
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

    def drop_mask(self, B: int, global_cols, n_cols):
        """Generate drop mask; [n_cols, B].
        1) generate global masks
        2) generate local masks
        3) resurrect random path in all-dead column
        4) concat global and local masks

        Args:
            - B: batch_size
            - global_cols: global columns which to alive [GB]
            - n_cols: the number of columns of mask
        """
        # global drop mask
        GB = global_cols.shape[0]
        # calc gdrop cols / samples
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]
        # gen gdrop mask
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.0

        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.0 - self.p_ldrop, [n_cols, LB]).astype(
            np.float32
        )
        alive_count = ldrop_mask.sum(axis=0)
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.0)[0]
        ldrop_mask[
            np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices
        ] = 1.0

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    def join(self, outs, global_cols):
        """
        Args:
            - outs: the outputs to join
            - global_cols: global drop path columns
        """
        n_cols = outs.size(0)
        out = outs

        if self.training:
            mask = self.drop_mask(out.size(1), global_cols, n_cols).to(
                out.device
            )  # [n_cols, B]
            mask = mask.view(*mask.size(), 1, 1, 1)  # unsqueeze to [n_cols, B, 1, 1, 1]
            n_alive = mask.sum(dim=0)  # [B, 1, 1, 1]
            masked_out = out * mask  # [n_cols, B, C, H, W]
            n_alive[n_alive == 0.0] = 1.0  # all-dead cases
            out = masked_out.sum(dim=0) / n_alive  # [B, C, H, W] / [B, 1, 1, 1]
        else:
            out = out.mean(dim=0)  # no drop

        return out

    def forward(self, x, global_cols, deepest=False):
        """
        global_cols works only in training mode.
        """
        x = self.doubler(x) if self.doubler else x
        outs = [x] * self.n_columns
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            n_depth_cols = self.n_columns - st
            cur_outs = torch.zeros(
                (
                    n_depth_cols,
                    x.size(0),
                    self.C_out,
                    x.size(2),
                    x.size(3),
                ),
                device=x.device,
            )
            if deepest:
                st = self.n_columns - 1  # last column only

            for col_idx, c in enumerate(range(st, self.n_columns)):
                cur_in = outs[c]  # current input
                cur_module = self.columns[c][i]  # current module
                cur_outs[col_idx] = cur_module(cur_in)

            # join
            # print("join in depth = {}, # of in_join = {}".format(i, len(cur_out)))
            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined

        if self.dropout_pos == "FD" and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1]  # for deepest case
