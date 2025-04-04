from logging import getLogger
import numpy as np
from PIL import Image
from cdatasets import get_dataset

import yaml


def rscript():
    log = getLogger("rscript")
    with open("./configs/base.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wrapper = get_dataset("standard", config, log)
    ds = wrapper.get_split("train")
    for x, y in ds:
        print(x.shape, y.shape)

    exit()

    rdir = "/mnt/cluster/nbl-datasets/fingervein/Foot_Print/Training/2 - Reference"
    fname = f"{rdir}/reference_data.npz"
    metadata = f"{rdir}/reference_metadata.csv"
    i = 0
    with open(metadata, "r") as f:
        for lin in f.readlines():
            print(lin)
            i += 1
            if i == 5:
                break

    # os.makedirs(f"{rdir}/frames_0", exist_ok=True)
    data = np.load(fname)
    frames = data["0"]
    print(frames.shape)
    exit()
    images = []
    for i in range(frames.shape[0]):
        frame = frames[i, :, :]
        frame = np.uint8(frame)
        img = Image.fromarray(frame)
        images.append(img)

    images[0].save(
        f"{rdir}/frames_0/test.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )


if __name__ == "__main__":
    rscript()
