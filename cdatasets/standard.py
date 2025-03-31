import itertools
from logging import Logger
from os.path import join as pjoin
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as A
from torch.utils.data import DataLoader

from utils import DatasetGenerator, Wrapper


class StandardWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
        **kwargs,
    ):
        """
        Standard wrapper for foot auth dataset
        """

        self.name = "standard"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.traindir = "./data/1 - Training/"
        self.classes = list(range(150))
        self.num_classes = len(self.classes)

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.prefetch_factor = config["prefetch_factor"]
        self.participant_IDs = np.arange(1, 151)
        self.speed_IDs = ["W1", "W2", "W3", "W4"]
        self.footwear_IDs = ["BF", "ST", "P1", "P2"]
        self.augmentation = A.Compose(
            [
                # self.resize,
                self.normalise,
            ]
        )

    def resize(self, x: torch.Tensor, size=(16, 32, 16)) -> torch.Tensor:
        x = F.interpolate(x.unsqueeze(0), size=size, mode="trilinear")
        return x.squeeze(0)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - x.min()) / (x.max() - x.min())
        if torch.isnan(x).any():
            x[torch.isnan(x)] = 0
        return x

    def init_metadata(self) -> pd.DataFrame:
        metadata_lst = []
        for participant_ID, footwear_ID, speed_ID in tqdm(
            itertools.product(self.participant_IDs, self.footwear_IDs, self.speed_IDs),
            desc="Loading metadata",
        ):
            metadata_path = pjoin(
                self.traindir,
                f"{participant_ID:03}",
                footwear_ID,
                speed_ID,
                "metadata.csv",
            )
            metadata_lst.append(pd.read_csv(metadata_path))

        metadata_train = pd.concat(metadata_lst).reset_index(drop=True)
        self.log.info("Metadata loaded.")
        self.log.info(f"Number of metadata files: {len(metadata_lst)}")

        return metadata_train

    def loop_splitset(self, ssplit: str) -> List[Any]:
        self.log.debug(f"Looping through splitset {ssplit}.")
        metadata = self.init_metadata()
        data = []
        for _, row in metadata.iterrows():
            participant_ID = row["ParticipantID"]
            side = row["Side"]
            footstep_ID = row["FootstepID"]
            footwear_ID = row["Footwear"]
            speed_ID = row["Speed"]
            sample_path = pjoin(
                self.traindir,
                f"{participant_ID:03}",
                str(footwear_ID),
                str(speed_ID),
                "pipeline_1.npz",
            )
            data.append((sample_path, f"{footstep_ID}", side, participant_ID - 1))

        return data

    def transform(self, datapoint: Iterable[Any]) -> Tuple:
        fname, footstep_ID, side, participant_ID = datapoint
        footsteps = np.load(fname)
        footstep = footsteps[f"{footstep_ID}"]

        # flip right footsteps along x axis
        if side == "Right":
            footstep = np.flip(footstep, axis=2)

        # add dimension for channel: shape (1,101,75,40) (channel,time,y,x)
        footstep = footstep[None, :, :, :]
        x = torch.from_numpy(footstep.astype(np.float32))
        labels = np.zeros(self.num_classes)
        labels[participant_ID - 1] = 1
        y = torch.tensor(labels)
        x = self.augmentation(x)
        x = x.squeeze(0)

        return x.float(), y.float()

    def get_split(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        self.log.debug("Looping through %s split." % split)
        data = self.loop_splitset(split)
        self.log.debug("Data-length for %s split: %d" % (split, len(data)))
        return DataLoader(
            DatasetGenerator(data, self.transform),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
            prefetch_factor=prefetch_factor or self.prefetch_factor,
            pin_memory=True,
            shuffle=True,
        )

    def augment(self, image: Any) -> Any:
        return image
