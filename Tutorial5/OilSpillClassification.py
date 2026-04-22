import os
import glob
import random
from datetime import datetime
from collections import Counter
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchcam.methods import SmoothGradCAMpp
from torch.utils.tensorboard import SummaryWriter
from satlaspretrain_models.utils import SatlasPretrain_weights
from satlaspretrain_models.model import Model as SatlasModel
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision import models as TorchModels
import pandas as pd


class OilSpillDataset(Dataset):
    def __init__(self, images_dir_oil: str, images_dir_lookalike: str, images_dir_clean: str, augment: bool,
                 input_size: int = 512, image_suffix: str = 'tif', merge_clean_lookalike=False):

        self.augment = augment
        image_paths_oil = []
        if images_dir_oil is not None:
            image_paths_oil = sorted(glob.glob(os.path.join(images_dir_oil, f"*.{image_suffix}")))
        image_paths_lookalike = []
        if images_dir_lookalike is not None:
            image_paths_lookalike = sorted(glob.glob(os.path.join(images_dir_lookalike, f"*.{image_suffix}")))
        image_paths_clean = []
        if images_dir_clean is not None:
            image_paths_clean = sorted(glob.glob(os.path.join(images_dir_clean, f"*.{image_suffix}")))

        if merge_clean_lookalike:
            image_paths_clean.extend(image_paths_lookalike)
            image_paths_lookalike = []

        self.images = []
        self.classes = []
        for img_path in image_paths_oil:
            self.images.append((img_path, torch.tensor([1],  dtype=torch.long)))
            self.classes.append(1)
        for img_path in image_paths_lookalike:
            self.images.append((img_path, torch.tensor([2])))
            self.classes.append(2)
        for img_path in image_paths_clean:
            self.images.append((img_path, torch.tensor([0])))
            self.classes.append(0)

        self.input_size = input_size
        self.low = -50.0
        self.high = 0.0
        self.no_data_in = 0.0
        self.no_data_out = 1.2

    def __len__(self):
        return len(self.images)

    def targets(self):
        return np.array(self.classes).astype(np.int64)

    def resize_image(self, img):
        if self.no_data_in is not None:
            nodata_mask = np.all(img == self.no_data_in, axis=0)
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
            if self.no_data_in is not None:
                resized_mask = cv2.resize(
                    nodata_mask.astype(np.uint8),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                resized[:, resized_mask] = self.no_data_in
            img = resized
        if new_h != self.input_size or new_w != self.input_size:
            pad_h = self.input_size - new_h
            pad_w = self.input_size - new_w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            padded = np.pad(
                img,
                pad_width=(
                    (0, 0),
                    (pad_top, pad_bottom),
                    (pad_left, pad_right)
                ),
                mode="constant",
                constant_values=self.no_data_in
            )
            img = padded
        assert img.shape == (C, self.input_size, self.input_size)

        return img

    def normalize_image(self, im_array):
        if self.no_data_in is not None:
            nodata_mask = np.all(im_array == self.no_data_in, axis=0)
            im_array[:, nodata_mask] = np.nan
        im_array = np.clip(im_array, self.low, self.high)
        im_array = (im_array - self.low) / (self.high - self.low) + 0.1
        im_array = np.nan_to_num(im_array, nan=self.no_data_out)

        return im_array

    def __getitem__(self, idx: int):
        img_path, class_id = self.images[idx]

        # --- read image ---
        with rasterio.open(img_path) as src:
            img = src.read()  # [C,H,W]

        if img.shape[0] > 2:
            img = img[:2, :, :]

        img = img[[1, 0], :, :] #  oil spill dataset is vv+vh while satlas pretrained expects vh+vv
        img = img.astype(np.float32)
        img = self.resize_image(img)
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

        img = torch.from_numpy(img)
        if self.augment:
            if random.random() < 0.2:
                img = torch.flip(img, dims=[2])
            if random.random() < 0.2:
                img = torch.flip(img, dims=[1])

        return img, class_id, img_path

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


def fetch_dataloaders(df_train, df_val=None, df_test=None, batch_size=4, augment_train=False, input_size=2048, merge_clean_lookalike=False):
    if df_train is not None:
        train_ds = OilSpillDataset(df_train['images_dir_oil'], df_train['images_dir_lookalike'], df_train['images_dir_clean'],
                                   augment=augment_train, input_size=input_size, merge_clean_lookalike=merge_clean_lookalike)
    if df_val is not None:
        val_ds = OilSpillDataset(df_val['images_dir_oil'], df_val['images_dir_lookalike'], df_val['images_dir_clean'],
                                 augment=False, input_size=input_size, merge_clean_lookalike=merge_clean_lookalike)
    if df_test is not None:
        test_ds = OilSpillDataset(df_test['images_dir_oil'], df_test['images_dir_lookalike'], df_test['images_dir_clean'],
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


class ResNet18Classifier(nn.Module):
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
        self.freezable_idx.extend([4, 5, 6, 7])

        # Channel sizes from ResNet-50
        # conv1: 64 ch
        # layer1: 64 ch (1/4)
        # layer2: 128 ch (1/8)
        # layer3: 256 ch (1/16)
        # layer4: 512 ch (1/32)

        self.backbone = nn.Sequential(
            resnet.conv1, # index 0
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 # index 7
        )
        c4 = 512

        # After conv, feature map is [B,c4,H/32,W/32] → avg-pool to [B, c4]
        # Then Linear to scalar logits
        # Following original implementation: https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet18
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_fc = nn.Linear(c4, num_classes)

        if freeze_backbone:
            self.freeze_backbone(partial=False)

    def freeze_backbone(self, partial=False):
        for idx in self.freezable_idx:
            for p in self.backbone[idx].parameters():
                p.requires_grad = False
        if partial:
            for p in self.backbone[self.freezable_idx[-1]].parameters():
                p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.avgpool(feats)
        feats = torch.flatten(feats, 1)
        return self.classifier_fc(feats)  #logits

    def extract_encoder_feats(self, x):
        feats = self.backbone(x)
        feats = feats.mean(dim=(2, 3))
        return feats

# This is based on SATLAS foundation model specifically pretrained on aerial images
# Reference to https://github.com/allenai/satlaspretrain_models/tree/main?tab=readme-ov-file#aerial-05-2mpx-high-res-imagery-pretrained-models
class SatlasClassifier(nn.Module):
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
        # x1: [B,128,,H/4,W/4], x2: [B,256,H/8,W/8], x3: [B,512,H/16,W/16], x4: [B,1024,H/32,W/32] (H/32)
        c4 = 1024

        # After conv, feature map is [B,1024,H/32,W/32] → pool to [B, 1024]
        # Then a residual layer and then linear to scalar logits
        self.adapter = torch.nn.Sequential(
                torch.nn.Conv2d(c4, c4, 3, padding=1),
                torch.nn.ReLU(inplace=True))
        self.classifier_fc = nn.Linear(c4, num_classes)

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
        feats = feats[-1]  # deepest feature map
        feats = self.adapter(feats)
        feats = torch.amax(feats, dim=(2, 3))
        return self.classifier_fc(feats)  # logits

    def extract_encoder_feats(self, x):
        feats = self.backbone(x)
        feats = feats[-1]
        feats = torch.amax(feats, dim=(2, 3))
        return feats


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
        try:
            print('Trying without classifier head!')
            ckpt_state = ckpt["model_state_dict"]
            ckpt_state.pop("classifier_fc.weight", None)
            ckpt_state.pop("classifier_fc.bias", None)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded weights from {model_path}")
        except:
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

def loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor):
    criterion = torch.nn.CrossEntropyLoss()
    if len(labels.shape) == 2:
        labels = labels.squeeze(1)
    loss = criterion(logits, labels)
    return loss


def print_trainable(model):
    # Just so we can be sure the right parameters are frozen
    for name, p in model.named_parameters():
        print(f"{name}: {'trainable' if p.requires_grad else 'frozen'}")


@torch.no_grad()
def update_confusion_matrix(cm, y_true, y_pred):
    y_true = y_true.view(-1).to(torch.int64)
    y_pred = y_pred.view(-1).to(torch.int64)
    num_classes = cm.shape[0]

    # Move to CPU as cm is in cpu
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    idx = y_true * num_classes + y_pred
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += binc.view(num_classes, num_classes)
    return cm


def pr_from_confusion_matrix(cm, eps=1e-12):
    cm = cm.to(torch.float64)

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / (cm.sum() + eps)
    return precision, recall, f1, acc


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
        device="cuda"
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
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
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
            for imgs, class_ids, _ in loader:
                imgs = imgs.to(device)
                class_ids = class_ids.to(device)
                logits = model(imgs)
                loss = loss_with_logits(logits, class_ids)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                preds = logits.argmax(dim=1)
                cm = update_confusion_matrix(cm, class_ids, preds)
                total_loss += loss.item() * class_ids.shape[0]
                total_n += class_ids.shape[0]
        pre, re, f1, acc = pr_from_confusion_matrix(cm)
        return total_loss / total_n, acc, pre, re, f1

    for epoch in range(start_epoch, num_epochs + 1):
        tr_loss, tr_acc, tr_prec, tr_re, tr_f1 = train_one_epoch(train_loader, train=True)
        va_loss, va_acc, va_prec, va_re, va_f1 = train_one_epoch(val_loader, train=False)

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

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_precision"].append(tr_prec)
        history["train_recall"].append(tr_re)
        history["train_f1"].append(tr_f1)

        # track best model by weighted overall F1
        va_f1_pos = va_f1[0]
        va_f1_count = 1
        for va in va_f1[1:]:
            va_f1_pos += 2.0 * va
            va_f1_count += 2
        va_f1_pos /= va_f1_count
        if va_f1_pos >= best_val:
            best_val = va_f1_pos
            save_checkpoint(path=os.path.join(ckpt_dir, f"best_model_{model_type}.pt"), model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch, best_val=best_val)
            print(f"With a weighted average validation F1-score of {best_val}, the best model got updated!")

        if save_model_every > 0:
             if epoch%save_model_every == 0:
                 save_checkpoint(path=os.path.join(ckpt_dir, f"model_e{epoch}_{model_type}.pt"), model=model, optimizer=optimizer,
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

    epochs = range(1, num_epochs + 1)
    plot_out_dir = os.path.join(run_dir, 'plots_'+run_name)
    os.makedirs(plot_out_dir, exist_ok=True)
    draw_figures(epochs, history, plot_out_dir, class_labels=class_labels, log_scale=False)
    draw_figures(epochs, history, plot_out_dir, class_labels=class_labels, log_scale=True)

    return history, run_name

def predict_probs(model, device, loader):
    model.to(device)
    model.eval()
    all_probs, all_y, all_paths = [], [], []
    with torch.no_grad():
        for imgs, class_ids, imgs_path in loader:
            imgs = imgs.to(device)
            class_ids = class_ids.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            if class_ids is not None:
                all_y.append(class_ids.detach().cpu().numpy())
            all_paths.extend(imgs_path)
    if all_y:
        all_y = np.concatenate(all_y)
    else:
        all_y = None
    return np.vstack(all_probs), all_paths, all_y

class SmoothGradCAMppSatlas(SmoothGradCAMpp):

    def _store_grad(self, grad, idx: int = 0) -> None:
        if self._hooks_enabled:
            self.hook_g[idx] = grad.data.permute(0, 3, 1, 2)

    def _hook_a(self, module: nn.Module, input, output, idx: int = 0) -> None:
        """Activation hook."""
        if self._hooks_enabled:
            self.hook_a[idx] = output.data.permute(0, 3, 1, 2)


def predict_probs_cam(model, device, loader, out_mask_dir, model_type):
    model.to(device)
    model.eval()
    if model_type == "satlas":
        cam_extractor = SmoothGradCAMppSatlas(model, target_layer=model.backbone.backbone.backbone.features[-3], std=0.01) #x3 (one before last in depth)
    else:
        cam_extractor = SmoothGradCAMpp(model, target_layer=model.backbone[6], std=0.01) #layer3 (one before last in depth)

    all_probs, all_y, all_paths = [], [], []
    for imgs, class_ids, imgs_path in loader:
        imgs = imgs.to(device)
        B = imgs.shape[0]
        imgs.requires_grad_(True)
        class_ids = class_ids.to(device)
        logits = model(imgs)
        class_idxs = logits.argmax(dim=1).detach().cpu().tolist()  # shape [B]
        cams_all = cam_extractor(class_idx=class_idxs, scores=logits)
        # Single target layer  -> cams is a list of length 1
        cams_all = cams_all[0]  # shape: [B, H/16, W/16]
        for b in range(B):
            cams = cams_all[b].detach().cpu().numpy()
            cams -= cams.min()
            cams /= (cams.max() + 1e-8)
            cams = to_pil_image(cams, mode='F')
            cmap = get_cmap("jet")
            # Resize mask and apply colormap
            if imgs_path[b].endswith('.tif'):
                with rasterio.open(imgs_path[b]) as src:
                    profile = src.profile
                    profile_out = profile.copy()
                    (H, W) = (src.height, src.width)
            else:
                with Image.open(imgs_path[b]) as src:
                    W, H = src.size

            cams = np.asarray(cams.resize((H, W), resample=Image.Resampling.BICUBIC))
            cams = (255 * cmap(np.asarray(cams))[:, :, :3]).astype(np.uint8)

            name = os.path.splitext(os.path.basename(imgs_path[b]))[0]
            if imgs_path[b].endswith('.tif'):
                profile_out.update(
                    {
                        "count": 3,
                        "dtype": rasterio.uint8,
                        "photometric": "RGB",
                    }
                )
                with rasterio.open(os.path.join(out_mask_dir, 'cam_' + name + '.tif'), "w", **profile_out) as dst:
                    # rasterio expects [C,H,W]
                    dst.write(cams.transpose(2, 0, 1))
            else:
                img = Image.fromarray(cams)  # mode inferred as RGB
                img.save(os.path.join(out_mask_dir, 'cam_' + name + '.png'))
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        if class_ids is not None:
            all_y.append(class_ids.detach().cpu().numpy())
        all_paths.extend(imgs_path)


    if all_y:
        all_y = np.concatenate(all_y)
    else:
        all_y = None
    return np.vstack(all_probs), all_paths, all_y

# Add a function to do simple inference from a trained model
def infer_on_geotiff(
    model: nn.Module,
    model_type: str,
    images_dir: str,
    out_mask_dir: str,
    device: torch.device,
    batch_size: int,
    input_size: int,
    num_classes: int):
    """
    Runs inference on (GeoTIFF) images.
    """
    model.eval()
    _,  _, loader = fetch_dataloaders(df_train=None,
                                        df_val=None,
                                        df_test={'images_dir_oil': images_dir, 'images_dir_lookalike': None, 'images_dir_clean': None},
                                        batch_size=batch_size,
                                        augment_train=False,
                                        input_size=input_size,
                                        merge_clean_lookalike=False if num_classes==3 else True)
    probs, paths, truths = predict_probs_cam(model, device, loader, out_mask_dir, model_type)
    return probs, paths, truths


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
                class_labels=None
                ):
    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device in use: {device}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)


    if model_type == "satlas":
        model = SatlasClassifier(freeze_backbone=freeze_backbone, num_classes=num_classes)
    else:
        model = ResNet18Classifier(freeze_backbone=freeze_backbone, num_classes=num_classes)

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
        device=device)

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
              class_labels=None):

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
        model = SatlasClassifier(freeze_backbone=True, num_classes=num_classes)
    else:
        model = ResNet18Classifier(freeze_backbone=True, num_classes=num_classes)

    _, _ = load_checkpoint(model_path=model_path, model=model, optimizer=None,
                           scheduler=None, map_location="cpu")

    for loader_name, loader in zip(["TRAIN", "VALIDATION", "TEST"], [train_loader, val_loader, test_loader]):
        print(f"-------------------- RESULTS ON {loader_name} DATASET")
        probs, paths, truths = predict_probs(model=model, loader=loader, device=device)
        preds = probs.argmax(axis=1)
        print("Overall Accuracy:", accuracy_score(truths, preds))
        print("Classification report")
        print(classification_report(truths, preds, target_names=class_labels))
        cm = confusion_matrix(truths, preds)
        print("Confusion matrix:\n", cm)
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_labels
        ).plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion matrix")
        plt.savefig(os.path.join(plot_out_dir, "CM_" + loader_name + ".png"))
        plt.close()

def main_infer(images_dir, model_path, out_masks_dir, model_type="satlas", batch_size=1, input_size=2048,
               num_classes=3, class_labels=None):
    if class_labels is None:
        class_labels = ["clean", "oil", "lookalike"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "satlas":
        model = SatlasClassifier(freeze_backbone=True, num_classes=num_classes)
    else:
        model = ResNet18Classifier(freeze_backbone=True, num_classes=num_classes)

    _, _ = load_checkpoint(model_path=model_path, model=model, optimizer=None,
                                            scheduler=None, map_location="cpu")
    model.to(device)
    os.makedirs(out_masks_dir, exist_ok=True)

    probs, paths, _ = infer_on_geotiff(
                        model=model,
                        model_type=model_type,
                        images_dir=images_dir,
                        out_mask_dir=out_masks_dir,
                        device=device,
                        batch_size=batch_size,
                        input_size=input_size,
                        num_classes=num_classes)
    y_preds = probs.argmax(axis=1)
    y_labels = [class_labels[y_preds[i]] for i in range(y_preds.shape[0])]

    data = {
        "image_path": paths,
        "predicted_class": y_labels,
    }
    for c in range(len(class_labels)):
        data["score_" + class_labels[c]]=[probs[i,c] for i in range(probs.shape[0])]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_masks_dir, "predictions.csv"), index=False)
