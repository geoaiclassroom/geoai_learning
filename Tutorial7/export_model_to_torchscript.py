"""
Export trained SegmentationUnetPlus model to TorchScript.

This script handles both local development (local geo_deep_learning module)
and installed package scenarios. It loads a checkpoint and converts it to 
TorchScript for deployment.

The exported model includes built-in normalization, so your inference library
only needs to pass raw image tensors and will get predictions directly.

Usage:
    python scripts/export_model_to_torchscript.py \
        --checkpoint path/to/model.ckpt \
        --output path/to/model.pt \
        --device cuda \
        --mean 0.5 0.5 0.5 \
        --std 0.2 0.2 0.2
"""

import sys
import logging
import importlib
from pathlib import Path
from typing import Optional

import torch
import lightning as L

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Add repo root to sys.path if running locally (allows local geo_deep_learning import)
repo_root = Path(__file__).parent.parent
if (repo_root / "geo_deep_learning").exists():
    sys.path.insert(0, str(repo_root))
    logger.info("Added repo root to sys.path: %s", repo_root)

# Now import from geo_deep_learning (will use local or installed version)
from geo_deep_learning.tasks_with_models.segmentation_unetplus import SegmentationUnetPlus


def register_legacy_module_aliases() -> None:
    """Register short module-path aliases used by older LightningCLI checkpoints."""
    alias_map = {
        "tasks_with_models": "geo_deep_learning.tasks_with_models",
        "tasks_with_models.segmentation_unetplus": (
            "geo_deep_learning.tasks_with_models.segmentation_unetplus"
        ),
        "tools": "geo_deep_learning.tools",
        "tools.callbacks": "geo_deep_learning.tools.callbacks",
        "utils": "geo_deep_learning.utils",
    }

    for alias, target in alias_map.items():
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(target)
        except ImportError:
            # Some aliases may not be required for a given checkpoint.
            continue


def export_to_torchscript(
    checkpoint_path: str,
    output_path: str,
    device: str = "cuda",
    use_tracing: bool = False,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> None:
    """
    Load a trained model checkpoint and export it to TorchScript.

    Args:
        checkpoint_path: Path to the .ckpt Lightning checkpoint file
        output_path: Path where the scripted model will be saved
        device: Device to load model on ("cuda" or "cpu")
        use_tracing: If True, use torch.jit.trace (requires dummy input).
                    If False, use torch.jit.script (more robust for control flow).
        mean: Normalization mean values (default: [0.5, 0.5, 0.5])
        std: Normalization std values (default: [0.2, 0.2, 0.2])
    """
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.2, 0.2, 0.2]
    
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    logger.info(f"Normalization - mean: {mean}, std: {std}")

    # Some checkpoints store short LightningCLI class paths (e.g. tasks_with_models.*).
    # Register aliases so these paths resolve both in local-dev and installed-package setups.
    register_legacy_module_aliases()
    
    # Set deterministic mode for reproducibility
    L.seed_everything(42, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the Lightning checkpoint
    # This will call configure_model() automatically and restore weights
    model = SegmentationUnetPlus.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on device: {device}")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Encoder: {model.encoder}")
    logger.info(f"Num classes: {model.num_classes}")
    logger.info(f"Image size: {model.image_size}")

    # Create a wrapper that includes normalization in the model
    # (strips away Lightning-specific code and handles preprocessing)
    class InferenceWrapper(torch.nn.Module):
        """Wrapper for inference-only usage with built-in normalization."""
        def __init__(self, lightning_model, mean: list, std: list):
            super().__init__()
            self.model = lightning_model.model  # Extract the actual smp.UnetPlusPlus
            
            # Register mean and std as buffers (saved with model, not trainable)
            self.register_buffer(
                "mean",
                torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
            )
            self.register_buffer(
                "std",
                torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with built-in normalization.
            
            Args:
                x: Raw image tensor of shape (B, 3, H, W) with values in [0, 255]
            
            Returns:
                Softmax probabilities of shape (B, num_classes, H, W)
            """
            # Step 1 - normalization: scale [0, 255] -> [0, 1]
            x = x / 255.0
            # Step 2 - standardization: (x - mean) / std
            x = (x - self.mean) / (self.std + 1e-6)
            logits = self.model(x)
            return torch.softmax(logits, dim=1)

    inference_model = InferenceWrapper(model, mean, std)
    inference_model = inference_model.to(device)
    inference_model.eval()

    # Create dummy input for tracing or direct scripting
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    logger.info("Converting to TorchScript...")
    
    if use_tracing:
        # Use torch.jit.trace (requires a dummy input to trace execution)
        logger.info("Using torch.jit.trace (execution-based)")
        scripted_model = torch.jit.trace(inference_model, dummy_input)
    else:
        # Use torch.jit.script (analyzes code structure, more robust)
        logger.info("Using torch.jit.script (AST-based)")
        try:
            scripted_model = torch.jit.script(inference_model)
        except Exception as e:
            logger.warning(f"jit.script failed: {e}. Falling back to jit.trace")
            scripted_model = torch.jit.trace(inference_model, dummy_input)

    # Test the scripted model with a dummy forward pass
    logger.info("Testing scripted model...")
    with torch.no_grad():
        dummy_output = scripted_model(dummy_input)
    logger.info(f"Dummy output shape: {dummy_output.shape}")

    # Save the scripted model
    logger.info(f"Saving scripted model to {output_path}")
    torch.jit.save(scripted_model, str(output_path))
    logger.info(f"✓ Model successfully exported to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1e6:.2f} MB")
    logger.info(f"✓ Normalization (mean={mean}, std={std}) is built into the model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export SegmentationUnetPlus checkpoint to TorchScript"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where TorchScript model will be saved (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load model on",
    )
    parser.add_argument(
        "--use-tracing",
        action="store_true",
        help="Use torch.jit.trace instead of torch.jit.script",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Per-channel normalization mean (default: 0.5 0.5 0.5)",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.2, 0.2, 0.2],
        help="Per-channel normalization std (default: 0.2 0.2 0.2)",
    )

    args = parser.parse_args()

    export_to_torchscript(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        use_tracing=args.use_tracing,
        mean=args.mean,
        std=args.std,
    )
