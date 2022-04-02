import glob
import os

for experiment in glob.glob("experiments/online/*"):
    for seed in glob.glob(f"{experiment}/*"):
        models = list(glob.glob(f"{seed}/model_*"))
        for model in models[::2]:
            os.remove(model)
