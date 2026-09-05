"""Hydra entrypoint used by torchrun for DDP training."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

# ``torchrun training/launch.py`` imports sibling training modules directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="td_fusion_8a100")
def main(cfg: DictConfig) -> None:
    Trainer(**cfg).run()


if __name__ == "__main__":
    main()
