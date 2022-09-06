import torch
import torch.nn as nn

from src.models.components.res_net import VideoSEResNet4


class LRWNet(nn.Module):
    """LRWNet Module."""

    def __init__(
        self,
        border=False,
        n_classes=500,
        se=False,
        in3d=1,
        out3d=64,
        rec_hidsize=768,
        rec_stack=2,
        rec_biderectional=False,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self.video_model = VideoSEResNet4(in3d=in3d, out3d=out3d, se=se)
        self.border = border

        if border:
            recurrent_in = (out3d * 8) + 1
        else:
            recurrent_in = out3d * 8

        if rec_biderectional:
            self.recurrent = nn.GRU(
                recurrent_in,
                rec_hidsize,
                num_layers=rec_stack,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
            self.classifier = nn.Linear(rec_hidsize * 2, n_classes)
        else:
            self.recurrent = nn.GRU(
                recurrent_in,
                rec_hidsize,
                num_layers=rec_stack,
                bidirectional=False,
                batch_first=True,
                dropout=dropout,
            )
            self.classifier = nn.Linear(rec_hidsize, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        self.recurrent.flatten_parameters()  # for faster computation on GPU

        vid_feat = self.video_model(x)
        vid_feat = self.dropout(vid_feat).float()

        if self.border:
            border = self.border[:, :, None]
            h, _ = self.recurrent(torch.cat((vid_feat, border), dim=-1))
        else:
            h, _ = self.recurrent(vid_feat)

        out = self.classifier(self.dropout(h)).mean(1)

        return out
