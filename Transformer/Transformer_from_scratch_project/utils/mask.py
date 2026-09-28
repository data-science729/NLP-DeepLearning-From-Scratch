import torch


class Transformermask:
    def __init__(self, pad_idx=1):
        self.pad_idx = pad_idx     # 记录填充符（Padding Token）在词表中的索引数字，默认值为 1。

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)    # src_mask形状[batch_size, 1, 1, src_len]。
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)    # tgt_pad_mask 的形状扩充后为：[batch_size, 1, 1, tgt_len]。
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        # 3. 按位与结合
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
