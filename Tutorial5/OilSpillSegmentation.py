import os
import glob
import copy
import random
from datetime import datetime
from collections import Counter
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from satlaspretrain_models.utils import SatlasPretrain_weights
from satlaspretrain_models.model import Model as SatlasModel
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision import models as TorchModels
import pandas as pd


class OilSpillDataset(Dataset):
    def __init__(self, images_dir_oil: str, images_dir_lookalike: str, images_dir_clean: str,
                 masks_dir_oil: str, masks_dir_lookalike: str, masks_dir_clean: str,
                 augment: bool, input_size: int = 512, image_suffix: str = 'tif', merge_clean_lookalike=False):

        self.augment = augment
        image_paths_oil = []
        mask_paths_oil = []
        if images_dir_oil is not None:
            image_paths_oil_temp = sorted(glob.glob(os.path.join(images_dir_oil, f"*.{image_suffix}")))
            if masks_dir_oil is not None:
                for img_path in image_paths_oil_temp:
                    filename = os.path.basename(img_path)
                    mask_path = os.path.join(masks_dir_oil, filename)
                    if  os.path.exists(mask_path):
                        image_paths_oil.append(img_path)
                        mask_paths_oil.append(mask_path)
            else:
                image_paths_oil = copy.deepcopy(image_paths_oil_temp)
                mask_paths_oil = [''] * len(image_paths_oil)

        image_paths_lookalike = []
        mask_paths_lookalike = []
        if images_dir_lookalike is not None:
            image_paths_lookalike_temp = sorted(glob.glob(os.path.join(images_dir_lookalike, f"*.{image_suffix}")))
            if masks_dir_lookalike is not None:
                for img_path in image_paths_lookalike_temp:
                    filename = os.path.basename(img_path)
                    mask_path = os.path.join(masks_dir_lookalike, filename)
                    if  os.path.exists(mask_path):
                        image_paths_lookalike.append(img_path)
                        mask_paths_lookalike.append(mask_path)
            else:
                image_paths_lookalike = copy.deepcopy(image_paths_lookalike_temp)
                mask_paths_lookalike = [''] * len(image_paths_lookalike)

        image_paths_clean = []
        mask_paths_clean = []
        if images_dir_clean is not None:
            image_paths_clean_temp = sorted(glob.glob(os.path.join(images_dir_clean, f"*.{image_suffix}")))
            if masks_dir_clean is not None:
                for img_path in image_paths_clean_temp:
                    filename = os.path.basename(img_path)
                    mask_path = os.path.join(masks_dir_clean, filename)
                    if  os.path.exists(mask_path):
                        image_paths_clean.append(img_path)
                        mask_paths_clean.append(mask_path)
            else:
                image_paths_clean = copy.deepcopy(image_paths_clean_temp)
                mask_paths_clean = [''] * len(image_paths_clean)

        if merge_clean_lookalike:
            image_paths_clean.extend(image_paths_lookalike)
            image_paths_lookalike = []
            mask_paths_clean.extend(mask_paths_lookalike)
            mask_paths_lookalike = []

        self.images = []
        self.classes = []
        for i, img_path in enumerate(image_paths_oil):
            self.images.append((img_path, mask_paths_oil[i], torch.tensor([1],  dtype=torch.long)))
            self.classes.append(1)
        for i, img_path in enumerate(image_paths_lookalike):
            self.images.append((img_path, mask_paths_lookalike[i], torch.tensor([2],  dtype=torch.long)))
            self.classes.append(2)
        for i, img_path in enumerate(image_paths_clean):
            self.images.append((img_path, mask_paths_clean[i], torch.tensor([0],  dtype=torch.long)))
            self.classes.append(0)

        self.input_size = input_size
        self.low = -50.0
        self.high = 0.0
        self.no_data_in = 0.0
        self.no_data_out = 1.2
        self.ignore = 255

    def __len__(self):
        return len(self.images)

    def targets(self):
        return np.array(self.classes).astype(np.int64)

    def resize_image(self, img, seg=None):
        if self.no_data_in is not None:
            nodata_mask = np.all(img == self.no_data_in, axis=0)
            if seg is not None:
                seg[nodata_mask] = self.ignore

        C, H, W = img.shape
        scale = min(self.input_size / H, self.input_size / W)
        if scale < 1.0:
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
        else:
            new_h = H
            new_w = W

        if new_h != H or new_w != W:
            resized = np.zeros((C, new_h, new_w), dtype=np.float32)
            for b in range(C):
                resized[b] = cv2.resize(
                    img[b],
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR
                )
            if seg is not None:
                resized_seg = cv2.resize(
                    seg,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                )
            if self.no_data_in is not None:
                resized_mask = cv2.resize(
                    nodata_mask.astype(np.uint8),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                resized[:, resized_mask] = self.no_data_in
                if seg is not None:
                    resized_seg[resized_mask] = self.ignore
            img = resized
            if seg is not None:
                seg = resized_seg
        if new_h != self.input_size or new_w != self.input_size:
            pad_h = self.input_size - new_h
            pad_w = self.input_size - new_w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            img = np.pad(
                img,
                pad_width=(
                    (0, 0),
                    (pad_top, pad_bottom),
                    (pad_left, pad_right)
                ),
                mode="constant",
                constant_values=self.no_data_in
            )
            if seg is not None:
                seg = np.pad(
                    seg,
                    pad_width=(
                        (pad_top, pad_bottom),
                        (pad_left, pad_right)
                    ),
                    mode="constant",
                    constant_values=self.ignore
                )
        assert img.shape == (C, self.input_size, self.input_size)
        if seg is not None:
            assert seg.shape == (self.input_size, self.input_size)

        return img, seg

    def normalize_image(self, im_array):
        if self.no_data_in is not None:
            nodata_mask = np.all(im_array == self.no_data_in, axis=0)
            im_array[:, nodata_mask] = np.nan
        im_array = np.clip(im_array, self.low, self.high)
        im_array = (im_array - self.low) / (self.high - self.low) + 0.1
        im_array = np.nan_to_num(im_array, nan=self.no_data_out)

        return im_array

    def __getitem__(self, idx: int):
        img_path, seg_path, class_id = self.images[idx]

        # --- read image ---
        with rasterio.open(img_path) as src:
            img = src.read()  # [C,H,W]

        if img.shape[0] > 2:
            img = img[:2, :, :]

        img = img[[1, 0], :, :] #  oil spill dataset is vv+vh while satlas pretrained expects vh+vv
        img = img.astype(np.float32)

        if seg_path:
            with rasterio.open(seg_path) as src:
                seg = src.read(1)  # [H,W]
                seg = seg.astype(np.uint8)
        else:
            seg = None

        img, seg = self.resize_image(img, seg)
        img  = self.normalize_image(img)

        # For debugging only
        # from rasterio.transform import Affine
        # with rasterio.open(
        #         "image.tif",
        #         "w",
        #         driver="GTiff",
        #         height=img.shape[1],
        #         width=img.shape[2],
        #         count=2,
        #         dtype="float32",
        #         transform=Affine.identity(),  # optional but explicit
        #         crs=None  # explicitly no CRS
        # ) as dst:
        #     dst.write(img)
        # with rasterio.open(
        #         "seg.tif",
        #         "w",
        #         driver="GTiff",
        #         height=seg.shape[1],
        #         width=seg.shape[2],
        #         count=1,
        #         dtype="uint8",
        #         transform=Affine.identity(),  # optional but explicit
        #         crs=None  # explicitly no CRS
        # ) as dst:
        #     dst.write(img)

        img = torch.from_numpy(img)
        if seg is not None:
            seg = torch.from_numpy(seg).long()
        if self.augment:
            if random.random() < 0.2:
                img = torch.flip(img, dims=[2])
                if seg is not None:
                    seg = torch.flip(seg, dims=[1])
            if random.random() < 0.2:
                img = torch.flip(img, dims=[1])
                if seg is not None:
                    seg = torch.flip(seg, dims=[0])

        if seg is None:
            seg = torch.zeros((img.shape[1], img.shape[2])).long()

        return img, seg, class_id, img_path, seg_path

    def generate_sampler(self):
        labels = torch.tensor(self.targets())
        class_counts = Counter(labels.tolist())
        if len(class_counts) == 2:
            negatives = class_counts[0]
        else:
            negatives = class_counts[0] + class_counts[2]
        if negatives > class_counts[1]:
            class_weights = {}
            for key in class_counts.keys():
                class_weights[key] = 1.0
            class_weights[1] = negatives / class_counts[1]
            sample_weights = torch.tensor(
                [class_weights[int(y)] for y in labels],
                dtype=torch.double
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return sampler, class_counts, class_weights
        else:
            return None, class_counts, None


def fetch_dataloaders(df_train, df_val=None, df_test=None, batch_size=4, augment_train=False, input_size=2048, merge_clean_lookalike=True):
    if df_train is not None:
        train_ds = OilSpillDataset(df_train['images_dir_oil'], df_train['images_dir_lookalike'], df_train['images_dir_clean'],
                                   df_train['masks_dir_oil'], df_train['masks_dir_lookalike'], df_train['masks_dir_clean'],
                                   augment=augment_train, input_size=input_size, merge_clean_lookalike=merge_clean_lookalike)
    if df_val is not None:
        val_ds = OilSpillDataset(df_val['images_dir_oil'], df_val['images_dir_lookalike'], df_val['images_dir_clean'],
                                 df_val['masks_dir_oil'], df_val['masks_dir_lookalike'], df_val['masks_dir_clean'],
                                 augment=False, input_size=input_size, merge_clean_lookalike=merge_clean_lookalike)
    if df_test is not None:
        test_ds = OilSpillDataset(df_test['images_dir_oil'], df_test['images_dir_lookalike'], df_test['images_dir_clean'],
                                  df_test['masks_dir_oil'], df_test['masks_dir_lookalike'], df_test['masks_dir_clean'],
                                  augment=False, input_size=input_size, merge_clean_lookalike=merge_clean_lookalike)

    if df_train is not None:
        sampler, class_counts, class_weights = train_ds.generate_sampler()
        print("train class counts:", class_counts)
        if sampler is not None:
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
            print("train class weights for weighted sampling:", class_weights)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        train_loader = None

    if df_val is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print("validation class counts:", Counter(val_ds.targets().tolist()))
    else:
        val_loader = None
    if df_test is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        print("test class counts:", Counter(test_ds.targets().tolist()))
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNet18UNetPP(nn.Module):
    def __init__(self, freeze_backbone=False, in_channels=2, num_classes=3):
        super().__init__()

        resnet = TorchModels.resnet18(weights=None)
        # Source: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        state_dict = torch.load(
            "./Pretrained/resnet18-f37072fd.pth",
            map_location="cpu"
        )
        resnet.load_state_dict(state_dict)

        self.freezable_idx = []
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        else:
            self.freezable_idx.append(0)
        self.freezable_idx.extend([1, 2, 3, 4])

        # Channel sizes from ResNet-18
        # conv0: 65 ch (1/2)
        # conv1: 64 ch (1/2)
        # layer1: 64 ch (1/4)
        # layer2: 128 ch (1/8)
        # layer3: 256 ch (1/16)
        # layer4: 512 ch (1/32)

        self.conv0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # /2

        self.pool = resnet.maxpool  # -> /4

        self.conv1 = resnet.layer1  # 64
        self.conv2 = resnet.layer2  # 128
        self.conv3 = resnet.layer3  # 256
        self.conv4 = resnet.layer4  # 512

        nb_filter = [64, 64, 128, 256, 512]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nested conv blocks
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], 64)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], 64)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], 128)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], 256)

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], 64)
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], 64)
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], 128)

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], 64)
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], 64)

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], 64)

        # Final layer(s)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        if freeze_backbone:
            self.freeze_backbone(partial=False)

    def freeze_backbone(self, partial=False):
        mods = [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4]
        for idx in self.freezable_idx:
            for p in mods[idx].parameters():
                p.requires_grad = False
        if partial:
            for p in mods[self.freezable_idx[-1]].parameters():
                p.requires_grad = True

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0(x)
        x1_0 = self.conv1(self.pool(x0_0))
        x2_0 = self.conv2(x1_0)
        x3_0 = self.conv3(x2_0)
        x4_0 = self.conv4(x3_0)

        # UNet++ dense skip connections
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        out = self.final(x0_4)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        return out

# This is based on SATLAS foundation model specifically pretrained on S1 imagery
# Reference to https://github.com/allenai/satlaspretrain_models#sentinel-1-pretrained-models
class SatlasUNet(nn.Module):
    def __init__(self, freeze_backbone=True, in_channels=2, num_classes=3):
        super().__init__()

        model_info = SatlasPretrain_weights["Sentinel1_SwinB_SI"]
        weights = torch.load("./Pretrained/sentinel1_swinb_si.pth", map_location=torch.device("cpu"))
        self.backbone = SatlasModel(
            model_info["num_channels"],
            model_info["multi_image"],
            model_info["backbone"],
            fpn=False,
            head=None,
            num_categories=None,
            weights=weights,
        )

        self.in_channels = in_channels
        if in_channels != 2:
            self.backbone.backbone.backbone.features[0][0] = torch.nn.Conv2d(in_channels,
                                                                    self.backbone.backbone.backbone.features[0][0].out_channels,
                                                                    kernel_size=(4, 4), stride=(4, 4))

        # Satlas feature sizes:
        # feats = self.backbone(x) -> [x1,x2,x3,x4] with
        # x1: [B,128,,H/4,W/4], x2: [B,256,H/8,W/8], x3: [B,512,H/16,W/16], x4: [B,1024,H/32,W/32]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Channels: [128, 256, 512, 1024]
        self.dec3 = ConvBlock(1024 + 512, 512)
        self.dec2 = ConvBlock(512 + 256, 256)
        self.dec1 = ConvBlock(256 + 128, 128)
        self.dec0 = ConvBlock(128, 64)

        self.up_final = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # ×2
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # ×2
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, 1)
        )

        if freeze_backbone:
            self.freeze_backbone(partial=False)

    def freeze_backbone(self, partial=False):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.in_channels != 2:
            for p in self.backbone.backbone.backbone.features[0][0].parameters():
                p.requires_grad = True
        if partial:
            for p in self.backbone.backbone.backbone.features[-1].parameters():
                p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        x1, x2, x3, x4 = feats

        d3 = self.dec3(torch.cat([self.up(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), x1], dim=1))
        d0 = self.dec0(self.up(d1))  # [B, 64, H/4, W/4]

        out = self.up_final(d0)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        return out

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    ckpt = {
        "epoch": epoch,  # next epoch to run (common convention)
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
        ckpt["scheduler_type"] = type(scheduler).__name__

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(model_path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(model_path, map_location=map_location, weights_only=True)
    missing = None
    unexpected = None
    try:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded weights from {model_path}")
    except Exception as e:
        print(e)
        print(f"Warning: continuing without loading the checkpoint from {model_path}!")

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 1)
    best_val = ckpt.get("best_val", float("-inf"))
    return start_epoch, best_val

# Now let's design our "loss" functions.
def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = 255,
    reduction: str = "mean",
):
    """
    Multi-class focal loss with ignore_index.

    logits: [B,C,H,W]
    targets: [B,H,W] with values in {0..C-1} or ignore_index
    """
    n, c, h, w = logits.shape
    logits = logits.permute(0, 2, 3, 1).reshape(-1, c)  # [N*C, C]
    targets = targets.view(-1)                          # [N*C]

    valid = targets != ignore_index
    logits = logits[valid]
    targets = targets[valid]

    if targets.numel() == 0:
        # no valid pixels in batch
        return logits.sum() * 0.0

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    targets_onehot = F.one_hot(targets, num_classes=c).float()
    p_t = (probs * targets_onehot).sum(dim=-1)
    p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
    alpha_t = alpha_t.sum(dim=-1)  # because only one non-zero per row

    ce = F.nll_loss(log_probs, targets, reduction="none")
    loss = alpha_t * (1 - p_t) ** gamma * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    reduction: str = "mean",
):
    """
    Multi-class Dice loss with ignore_index.

    logits:  [B, C, H, W]
    targets: [B, H, W] with values in {0..C-1} or ignore_index
    """
    B, C, H, W = logits.shape

    # Convert targets → one-hot
    valid_mask = targets != ignore_index

    if valid_mask.sum() == 0:
        return logits.sum() * 0.0

    # Temporarily set ignored pixels to 0 so one_hot works
    targets = targets.clone()
    targets[~valid_mask] = 0

    targets_onehot = F.one_hot(targets, num_classes=C)  # [B,H,W,C]
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

    # Probabilities
    probs = F.softmax(logits, dim=1)

    # Apply mask
    valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]
    probs = probs * valid_mask
    targets_onehot = targets_onehot * valid_mask

    # Compute Dice per class
    dims = (0, 2, 3)

    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)

    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)

    # Dice loss = 1 - Dice score
    loss = 1.0 - dice  # [C]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def print_trainable(model):
    # Just so we can be sure the right parameters are frozen
    for name, p in model.named_parameters():
        print(f"{name}: {'trainable' if p.requires_grad else 'frozen'}")


@torch.no_grad()
def update_confusion_matrix(cm, y_true, y_pred, ignore_index):
    y_true = y_true.view(-1).to(torch.int64)
    y_pred = y_pred.view(-1).to(torch.int64)
    num_classes = cm.shape[0]

    # Move to CPU as cm is in cpu
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    valid = y_true != ignore_index
    y_true = y_true[valid]

    if y_true.numel() == 0:
        return cm, torch.zeros((num_classes, num_classes))  # nothing to update

    y_pred = y_pred[valid]

    idx = y_true * num_classes + y_pred
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    single_cm = binc.view(num_classes, num_classes)
    cm += single_cm
    return cm, single_cm


def pr_from_confusion_matrix(cm, eps=1e-12, ignore_background=False):
    cm = cm.to(torch.float64)

    tp = torch.diag(cm) #intersection
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    union = cm.sum(dim=1) + cm.sum(dim=0) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / (cm.sum() + eps)

    iou = tp / (union + eps)

    dice = 2 * tp / (2 * tp + fp + fn + eps)

    if ignore_background:
        precision = precision[1:]
        recall = recall[1:]
        f1 = f1[1:]
        iou = iou[1:]
        dice = dice[1:]

    return precision, recall, f1, iou, dice, acc


def draw_figures(epochs, history, save_folder, class_labels, log_scale=False):
    for metric_key in history.keys():
        if 'train' in metric_key:
            metric_key_val = metric_key.replace("train", "val")
            history_metric_value = history[metric_key]
            history_metric_value_val =  history[metric_key_val]
            if isinstance(history_metric_value[0], float):
                history_metric_values = np.zeros((len(history_metric_value), 1))
                history_metric_values_val = np.zeros((len(history_metric_value_val), 1))
                history_metric_values[:, 0] =  np.array([v for v in history_metric_value])
                history_metric_values_val[:, 0] =  np.array([v for v in history_metric_value_val])
            elif history_metric_value[0].ndim == 0:
                history_metric_values = np.zeros((len(history_metric_value), 1))
                history_metric_values_val = np.zeros((len(history_metric_value_val), 1))
                history_metric_values[:, 0] = np.array([v.item() for v in history_metric_value])
                history_metric_values_val[:, 0] = np.array([v.item() for v in history_metric_value_val])
            else:
                history_metric_values = np.zeros((len(history_metric_value), history_metric_value[0].shape[0]))
                history_metric_values_val = np.zeros((len(history_metric_value_val), history_metric_value[0].shape[0]))
                for k in range(history_metric_value[0].shape[0]):
                    history_metric_values[:, k] = np.array([v[k].item() for v in history_metric_value])
                    history_metric_values_val[:, k] = np.array([v[k].item() for v in history_metric_value_val])

            for k in range(history_metric_values.shape[1]):
                plt.figure(figsize=(8, 5))
                plt.plot(
                    epochs,
                    history_metric_values[:, k],
                    label=metric_key.replace("_", " "),
                    color="blue",
                    linestyle="--",
                    linewidth=2
                )
                plt.plot(
                    epochs,
                    history_metric_values_val[:, k],
                    label=metric_key_val.replace("_", " "),
                    color="red",
                    linestyle="-",
                    linewidth=2
                )
                if log_scale:
                    plt.yscale("log")
                plt.xlabel("Epoch")
                plt.ylabel(metric_key.replace("train_", ""))
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                name_term = ""
                if log_scale:
                    name_term = "_log"
                class_name = ""
                if history_metric_values.shape[1] > 1:
                    class_name = class_labels[k] + "_"
                plt.savefig(os.path.join(save_folder, class_name + metric_key.replace("train_", "") + name_term + "_curve.png"))
                plt.close()

# Let's manage the whole training process
def run_training(
        ckpt_dir: str,
        run_dir: str,
        model: nn.Module,
        model_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
        freeze_backbone_partial: bool = False,
        use_scheduler: bool = True,
        num_classes: int = 3,
        class_labels=None,
        save_model_every: int = 0,
        resume_training: bool = False,
        resume_weights_only: bool = False,
        resume_ckpt_path: str = "",
        device="cuda",
        dice_factor=0.5,
        focal_factor=0.5,
        ignore_index=255
):
    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]
    model.to(device)
    model.train()
    if freeze_backbone:
        model.freeze_backbone(partial=freeze_backbone_partial)

    print_trainable(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        scheduler = None

    run_name = (
        f"oil_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    if resume_training:
        if resume_weights_only:
            _, _ = load_checkpoint(model_path=resume_ckpt_path, model=model, optimizer=None,
                                                scheduler=None, map_location="cpu")
            start_epoch = 1
            best_val = 0.0
        else:
            start_epoch, best_val = load_checkpoint(model_path=resume_ckpt_path, model=model, optimizer=optimizer,
                                                    scheduler=scheduler, map_location="cpu")
        model.to(device)
    else:
        start_epoch = 1
        best_val = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_precision": [],
        "train_recall": [],
        "train_iou": [],
        "train_dice": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_iou": [],
        "val_dice": [],
    }

    def train_one_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        if freeze_backbone:
            model.freeze_backbone(partial=freeze_backbone_partial)

        total_loss, total_n = 0.0, 0
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        with torch.set_grad_enabled(train):
            for imgs, targets, _, _, _ in loader: #img, seg, class_id, img_path, seg_path
                imgs = imgs.to(device)
                targets = targets.to(device)
                logits = model(imgs) #[B, C, H, W]

                focal_loss = multiclass_focal_loss(logits, targets)
                dice_loss = multiclass_dice_loss(logits, targets)
                loss = dice_factor * dice_loss + focal_factor * focal_loss
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                preds = logits.argmax(dim=1)
                cm, _ = update_confusion_matrix(cm, targets, preds, ignore_index=ignore_index)
                total_loss += loss.item() * targets.shape[0]
                total_n += targets.shape[0]
        pre, re, f1, iou, dice, acc = pr_from_confusion_matrix(cm)
        return total_loss / total_n, acc, pre, re, f1, iou, dice

    for epoch in range(start_epoch, num_epochs + 1):
        tr_loss, tr_acc, tr_prec, tr_re, tr_f1, tr_iou, tr_dice = train_one_epoch(train_loader, train=True)
        va_loss, va_acc, va_prec, va_re, va_f1, va_iou, va_dice = train_one_epoch(val_loader, train=False)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d} | lr={lr:.2e} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        # ---- store in history ----
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_precision"].append(va_prec)
        history["val_recall"].append(va_re)
        history["val_f1"].append(va_f1)
        history["val_iou"].append(va_iou)
        history["val_dice"].append(va_dice)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_precision"].append(tr_prec)
        history["train_recall"].append(tr_re)
        history["train_f1"].append(tr_f1)
        history["train_iou"].append(tr_iou)
        history["train_dice"].append(tr_dice)

        # track best model by weighted overall iou
        va_iou_pos = va_iou[0]
        va_iou_count = 1
        for va in va_iou[1:]:
            va_iou_pos += 2.0 * va
            va_iou_count += 2
        va_iou_pos /= va_iou_count
        if va_iou_pos >= best_val:
            best_val = va_iou_pos
            save_checkpoint(path=os.path.join(ckpt_dir, f"best_seg_model_{model_type}.pt"), model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, best_val=best_val)
            print(f"With a weighted average validation IoU of {best_val}, the best model got updated!")

        if save_model_every > 0:
             if epoch%save_model_every == 0:
                 save_checkpoint(path=os.path.join(ckpt_dir, f"model_seg_e{epoch}_{model_type}.pt"), model=model, optimizer=optimizer,
                                 scheduler=scheduler, epoch=epoch, best_val=best_val)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", va_acc, epoch)
        writer.add_scalar("LR", lr, epoch)
        for c in range(num_classes):
            class_name = class_labels[c]
            writer.add_scalar(f"Precision/train-{class_name}", tr_prec[c].item(), epoch)
            writer.add_scalar(f"Precision/val-{class_name}", va_prec[c].item(), epoch)
            writer.add_scalar(f"Recall/train-{class_name}", tr_re[c].item(), epoch)
            writer.add_scalar(f"Recall/val-{class_name}", va_re[c].item(), epoch)
            writer.add_scalar(f"F1Score/train-{class_name}", tr_f1[c].item(), epoch)
            writer.add_scalar(f"F1Score/val-{class_name}", va_f1[c].item(), epoch)
            writer.add_scalar(f"IoU/train-{class_name}", tr_iou[c].item(), epoch)
            writer.add_scalar(f"IoU/val-{class_name}", va_iou[c].item(), epoch)
            writer.add_scalar(f"Dice/train-{class_name}", tr_dice[c].item(), epoch)
            writer.add_scalar(f"Dice/val-{class_name}", va_dice[c].item(), epoch)

    epochs = range(1, num_epochs + 1)
    plot_out_dir = os.path.join(run_dir, 'plots_'+run_name)
    os.makedirs(plot_out_dir, exist_ok=True)
    draw_figures(epochs, history, plot_out_dir, class_labels=class_labels, log_scale=False)
    draw_figures(epochs, history, plot_out_dir, class_labels=class_labels, log_scale=True)

    return history, run_name

def predict_masks(model, device, loader, out_mask_dir, num_classes, save_seg=True, ignore_index=255, eval=False):
    model.to(device)
    model.eval()
    all_paths = []
    total_loss, total_n = 0.0, 0
    all_pre, all_re, all_f1, all_iou, all_dice, all_acc, all_loss = [], [], [], [], [], [], []
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for imgs, targets, _, imgs_path, seg_path in loader: #img, seg, class_id, img_path, seg_path
            imgs = imgs.to(device)
            B = imgs.shape[0]
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            if eval:
                for b in range(B):
                    target = targets[b:b + 1, ...].to(device)
                    focal_loss = multiclass_focal_loss(logits[b:b + 1, ...], target)
                    dice_loss = multiclass_dice_loss(logits[b:b + 1, ...], target)
                    loss = 0.5 * dice_loss + 0.5 * focal_loss
                    cm, single_cm = update_confusion_matrix(cm, target, preds[b:b + 1, ...], ignore_index=ignore_index)
                    loss_item = loss.item()
                    total_loss += loss_item * target.shape[0]
                    total_n += target.shape[0]
                    pre, re, f1, iou, dice, acc = pr_from_confusion_matrix(single_cm)
                    all_pre.append(pre)
                    all_re.append(re)
                    all_f1.append(f1)
                    all_iou.append(iou)
                    all_dice.append(dice)
                    all_acc.append(acc)
                    all_loss.append(loss_item)
            if save_seg:
                for b in range(B):
                    # Resize mask and apply colormap
                    if imgs_path[b].endswith('.tif'):
                        with rasterio.open(imgs_path[b]) as src:
                            profile = src.profile
                            profile_out = profile.copy()
                            H, W = src.height, src.width
                    else:
                        with Image.open(imgs_path[b]) as src:
                            W, H = src.size
                    pred = preds[b,...].detach().cpu().numpy().astype(np.uint8)
                    pred = np.asarray(Image.fromarray(pred).resize((W, H), resample=Image.Resampling.NEAREST))
                    name = os.path.splitext(os.path.basename(imgs_path[b]))[0]
                    if imgs_path[b].endswith('.tif'):
                        profile_out.update(
                            {
                                "count": 1,
                                "dtype": rasterio.uint8
                            }
                        )
                        with rasterio.open(os.path.join(out_mask_dir, 'seg_' + name + '.tif'), "w", **profile_out) as dst:
                            # rasterio expects [H,W]
                            dst.write(pred, 1)  # shape [H, W]
                    else:
                        img = Image.fromarray(preds[b,:,:])
                        img.save(os.path.join(out_mask_dir, 'seg_' + name + '.png'))
            all_paths.extend(imgs_path)
    if total_n > 0:
        pre, re, f1, iou, dice, acc = pr_from_confusion_matrix(cm)
        loss = total_loss / total_n
    else:
        pre, re, f1, iou, dice, acc, loss = None, None, None, None, None, None, None
    return all_paths, all_pre, all_re, all_f1, all_iou, all_dice, all_acc, all_loss, pre, re, f1, iou, dice, acc, loss, cm

# Add a function to do simple inference from a trained model
def infer_on_geotiff(
    model: nn.Module,
    model_type: str,
    images_dir: str,
    out_mask_dir: str,
    device: torch.device,
    batch_size: int,
    input_size: int,
    num_classes: int,
    ignore_index: int):
    """
    Runs inference on (GeoTIFF) images.
    """
    model.eval()
    _,  _, loader = fetch_dataloaders(df_train=None,
                                        df_val=None,
                                        df_test={'images_dir_oil': images_dir, 'images_dir_lookalike': None, 'images_dir_clean': None,
                                                 'masks_dir_oil': None, 'masks_dir_lookalike': None, 'masks_dir_clean': None},
                                        batch_size=batch_size,
                                        augment_train=False,
                                        input_size=input_size,
                                        merge_clean_lookalike=False if num_classes==3 else True)
    predict_masks(model, device, loader, out_mask_dir,
                  num_classes=num_classes, save_seg=True,
                  ignore_index=ignore_index, eval=False)



def main_train(train_df: dict,
                val_df: dict,
                test_df: dict,
                ckpt_dir: str,
                run_dir: str,
                model_type="satlas",
                augment_train: bool = False,
                num_epochs: int = 30,
                batch_size: int = 4,
                lr: float = 3e-3,
                weight_decay: float = 1e-4,
                freeze_backbone: bool = True,
                freeze_backbone_partial: bool = False,
                use_scheduler: bool = True,
                save_model_every: int = 0,
                resume_training: bool = False,
                resume_weights_only: bool = False,
                resume_ckpt_path: str = "",
                input_size: int = 2048,
                num_classes: int = 3,
                class_labels=None,
                dice_factor=0.5,
                focal_factor=0.5,
                ignore_index=255
                ):
    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device in use: {device}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)


    if model_type == "satlas":
        model = SatlasUNet(freeze_backbone=freeze_backbone, num_classes=num_classes)
    else:
        model = ResNet18UNetPP(freeze_backbone=freeze_backbone, num_classes=num_classes)

    train_loader, val_loader, test_loader = fetch_dataloaders(df_train=train_df, df_val=val_df, df_test=test_df,
                                                              batch_size=batch_size,
                                                              augment_train=augment_train, input_size=input_size,
                                                              merge_clean_lookalike=False if num_classes==3 else True)

    print("Training started ... ")
    _, run_name = run_training(ckpt_dir = ckpt_dir,
        run_dir=run_dir,
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        freeze_backbone_partial=freeze_backbone_partial,
        use_scheduler=use_scheduler,
        num_classes=num_classes,
        class_labels=class_labels,
        save_model_every=save_model_every,
        resume_training=resume_training,
        resume_weights_only=resume_weights_only,
        resume_ckpt_path=resume_ckpt_path,
        device=device,
        dice_factor=dice_factor,
        focal_factor=focal_factor,
        ignore_index=ignore_index
        )

    return run_name

def main_test(train_df: dict,
              val_df: dict,
              test_df: dict,
              model_path: str,
              plot_out_dir: str,
              model_type="satlas",
              batch_size: int = 4,
              input_size: int = 2048,
              num_classes: int = 3,
              class_labels=None,
              save_seg=False,
              ignore_index=255):

    _, _, test_loader = fetch_dataloaders(df_train=None, df_val=None, df_test=test_df,
                                          batch_size=batch_size,
                                          augment_train=False, input_size=input_size,
                                          merge_clean_lookalike=False if num_classes == 3 else True)

    _, _, val_loader = fetch_dataloaders(df_train=None, df_val=None, df_test=val_df,
                                          batch_size=batch_size,
                                          augment_train=False, input_size=input_size,
                                          merge_clean_lookalike=False if num_classes == 3 else True)

    _, _, train_loader = fetch_dataloaders(df_train=None, df_val=None, df_test=train_df,
                                           batch_size=batch_size,
                                           augment_train=False, input_size=input_size,
                                           merge_clean_lookalike=False if num_classes == 3 else True)

    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "satlas":
        model = SatlasUNet(freeze_backbone=True, num_classes=num_classes)
    else:
        model = ResNet18UNetPP(freeze_backbone=True, num_classes=num_classes)

    _, _ = load_checkpoint(model_path=model_path, model=model, optimizer=None,
                           scheduler=None, map_location="cpu")

    for loader_name, loader in zip(["TRAIN", "VALIDATION", "TEST"], [train_loader, val_loader, test_loader]):
        print(f"-------------------- RESULTS ON {loader_name} DATASET")
        (all_paths, all_pre, all_re, all_f1, all_iou, all_dice,
         all_acc, all_loss, pre, re, f1, iou, dice, acc, loss, cm) = predict_masks(model=model, loader=loader, device=device,
                                                                         save_seg=save_seg, num_classes=num_classes,
                                                                         out_mask_dir=plot_out_dir, ignore_index=ignore_index,
                                                                         eval=True)
        print(f"Overall Accuracy: {acc:.3f}")
        print(f"Overall Loss: {loss:.5f}")
        for c in range(num_classes):
            print(f'Metrics of class {class_labels[c]}')
            print(f"    Precision: {pre[c]:.3f}")
            print(f"    Recall: {re[c]:.3f}")
            print(f"    F1 Score: {f1[c]:.3f}")
            print(f"    IoU: {iou[c]:.3f}")
            print(f"    Dice Score: {dice[c]:.3f}")

        print("Confusion matrix:\n", cm)

        cm = cm.cpu().numpy().astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # prevent division by zero
        cm_norm = cm / row_sums

        ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=class_labels
        ).plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.title("Row-Normalized Confusion matrix")
        plt.savefig(os.path.join(plot_out_dir, "CM_SEG_" + loader_name + ".png"))
        plt.close()

        data = {
            "image_path": all_paths,
        }
        if all_pre:
            for c in range(len(class_labels)):
                data["Precision_" + class_labels[c]] = [all_pre[i][c].item() for i in range(len(all_paths))]
                data["Recall_" + class_labels[c]] = [all_re[i][c].item() for i in range(len(all_paths))]
                data["IoU_" + class_labels[c]] = [all_iou[i][c].item() for i in range(len(all_paths))]
                data["F1_" + class_labels[c]] = [all_f1[i][c].item() for i in range(len(all_paths))]
                data["Dice_" + class_labels[c]] = [all_f1[i][c].item() for i in range(len(all_paths))]
            data["Loss"] = [all_loss[i] for i in range(len(all_paths))]
            data["Accuracy"] = [all_acc[i].item() for i in range(len(all_paths))]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(plot_out_dir, loader_name + "_predictions.csv"), float_format="%.3f", index=False)

def main_infer(images_dir, model_path, out_masks_dir, model_type="satlas", batch_size=1, input_size=2048,
               num_classes=3, class_labels=None, ignore_index=255):
    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "satlas":
        model = SatlasUNet(freeze_backbone=True, num_classes=num_classes)
    else:
        model = ResNet18UNetPP(freeze_backbone=True, num_classes=num_classes)

    _, _ = load_checkpoint(model_path=model_path, model=model, optimizer=None,
                                            scheduler=None, map_location="cpu")
    model.to(device)
    os.makedirs(out_masks_dir, exist_ok=True)

    infer_on_geotiff(model=model,
                      model_type=model_type,
                      images_dir=images_dir,
                      out_mask_dir=out_masks_dir,
                      device=device,
                      batch_size=batch_size,
                      input_size=input_size,
                      num_classes=num_classes,
                      ignore_index=ignore_index)

