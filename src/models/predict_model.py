# # such that it takes a pre-trained model file and creates prediction for some data.
# # Recommended interface is that users can give this file either a folder with raw images
# # that gets loaded in or a numpy or pickle file with already loaded images e.g. something like this

# # python src/models/predict_model.py
# python src/models/trained_model.pt   # file containing a pretrained model
# data/example_images.npz

import argparse

import numpy as np
import torch

from src.models.model import MyAwesomeModel


def predict() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("data_to_predict", type=str)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.to(device)

    imgs = np.load(args.data_to_predict)
    imgs = torch.tensor(
        imgs["images"].reshape(-1, 1, 28, 28), dtype=torch.float, device=device
    )  # need to load only images and not labels

    log_probs = model(imgs)
    prediction = log_probs.argmax(dim=-1)
    probs = log_probs.softmax(dim=-1)

    print("Predictions")
    for i in range(imgs.shape[0]):
        print(
            f"Image {i+1} predicted to be class {prediction[i].item()} with probability {probs[i, prediction[i]].item()}"
        )


if __name__ == "__main__":
    predict()
