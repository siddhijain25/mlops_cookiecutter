from src.data.make_dataset import CorruptMnist
import torch
import os.path
import pytest


# dataset = MNIST(...)
# assert len(dataset) == N_train for training and N_test for test
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def load_data():
    # dataset = MNIST
    train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder=None)
    test_set = CorruptMnist(train=False, in_folder="data/raw", out_folder=None)
    return train_set, test_set

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_data():
    train_set, test_set = load_data()
    assert len(train_set) == 40000, "Train dataset did not have the correct number of samples"
    assert len(test_set) == 5000,  "Test dataset did not have the correct number of samples"
    # assert shape

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_datapoint_shape():
    train_set, test_set = load_data()
    assert train_set[0][0].shape == torch.Size([1,28,28]), "Train datapoint did not have correct shape"
    assert test_set[0][0].shape == torch.Size([1,28,28]), "Test datapoint did not have correct shape"

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_label_rep():
    train_set, test_set = load_data()
    torch.unique(train_set.targets) == 9
    # assert torch.unique(test_set.targets) == 9

train_set, test_set = load_data()
@pytest.mark.parametrize("test_input,expected", [("len(train_set)", 40000), ("len(test_set)", 5000)])
def test_parametrize_eval(test_input, expected):
    assert eval(test_input) == expected