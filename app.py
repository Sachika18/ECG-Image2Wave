import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import os

# --------------------------
# ResNet-based encoder + UNet 1D decoder (training architecture)
# --------------------------
class ResNetPanelEncoder(nn.Module):
    def __init__(self, out_dim=512, pretrained=False):
        super().__init__()
        rn = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        f = self.features(x)
        g = self.pool(f).view(f.size(0), -1)
        return self.proj(g)


class UNet1DDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_T=10250, base_ch=128, up_steps=5):
        super().__init__()
        self.out_T = out_T
        L0 = math.ceil(out_T / (2 ** up_steps))
        self.project = nn.Linear(latent_dim, base_ch * L0)

        layers = []
        in_ch = base_ch
        for i in range(up_steps):
            out_ch = max(base_ch // (2 ** (i+1)), 16)
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(True))
            in_ch = out_ch
        self.up = nn.Sequential(*layers)

        self.final = nn.Conv1d(in_ch, 12, 3, padding=1)

    def forward(self, z):
        B = z.size(0)
        # project to (B, base_ch * L0) then reshape to (B, base_ch, L0)
        x = self.project(z)
        # infer L0 from project out_features and base channel 128
        L0 = self.project.out_features // 128
        x = x.view(B, 128, L0)
        x = self.up(x)
        x = self.final(x)
        if x.size(2) >= self.out_T:
            return x[:, :, :self.out_T]
        else:
            pad = torch.zeros(B, 12, self.out_T - x.size(2), device=x.device)
            return torch.cat([x, pad], dim=2)


class ResNetUNetDigitizer(nn.Module):
    def __init__(self, out_T=10250, enc_dim=512, fuse_dim=256):
        super().__init__()
        self.encoder = ResNetPanelEncoder(out_dim=enc_dim, pretrained=False)
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
        )
        self.decoder = UNet1DDecoder(latent_dim=fuse_dim, out_T=out_T)

    def forward(self, panels):
        B, P, C, H, W = panels.shape
        feats = self.encoder(panels.view(B*P, C, H, W)).view(B, P, -1)
        z = self.fuse(feats.mean(dim=1))
        return self.decoder(z)




# --------------------------
# Predictor Class
# --------------------------
class ECGPredictor:
    def __init__(self, model_path, max_points=10250):

        self.device = "cpu"

        self.model = ResNetUNetDigitizer(out_T=max_points).to(self.device)
        # Load checkpoint safely: accept checkpoints from different architectures by
        # only copying parameters that both exist in the current model and have
        # matching shapes. This prevents crashes when a checkpoint was saved
        # from a different encoder/decoder (e.g., ResNet features).
        try:
            ck = torch.load(model_path, map_location=self.device)
            # support checkpoints that wrap state_dict inside a dict (e.g. {'state_dict': ...})
            if isinstance(ck, dict) and 'state_dict' in ck:
                ck = ck['state_dict']

            model_dict = self.model.state_dict()
            filtered_ck = {}
            skipped = []

            for k, v in ck.items():
                # remove common DataParallel/Lightning prefixes
                if k.startswith('module.'):
                    k2 = k[len('module.'):]
                else:
                    k2 = k

                if k2 in model_dict:
                    if v.shape == model_dict[k2].shape:
                        filtered_ck[k2] = v
                    else:
                        skipped.append((k2, v.shape, model_dict[k2].shape))
                else:
                    skipped.append((k2, None, None))

            # load the matched params
            self.model.load_state_dict(filtered_ck, strict=False)

            if skipped:
                print("Warning: some checkpoint keys were skipped when loading model:")
                for item in skipped[:20]:
                    if item[1] is None:
                        print(f" - missing in model: {item[0]}")
                    else:
                        print(f" - shape mismatch: {item[0]} checkpoint{item[1]} != model{item[2]}")
                if len(skipped) > 20:
                    print(f" - ... (+{len(skipped)-20} more)")

        except Exception as e:
            # If loading fails completely, keep the randomly-initialized model
            # but surface the error to the console so the Streamlit process can
            # show logs and the UI won't stay blank due to an exception.
            print(f"Error loading checkpoint '{model_path}': {e}")
        self.model.eval()

        self.max_panels = 12
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def load_panels(self, paths):
        imgs = []

        for p in paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        # Pad to 12 panels
        while len(imgs) < self.max_panels:
            imgs.append(torch.zeros(3, 128, 128))

        imgs = imgs[:12]

        panels = torch.stack(imgs, dim=0)  # (12,3,128,128)
        return panels.unsqueeze(0)         # (1,12,3,128,128)

    def predict(self, panel_paths):
        x = self.load_panels(panel_paths).to(self.device)

        with torch.no_grad():
            out = self.model(x)

        return out.squeeze(0).cpu().numpy()     # (12, 10250)
