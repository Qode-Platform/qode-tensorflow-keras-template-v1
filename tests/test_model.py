import numpy as np

from src.model import build_model


def test_output_shape():
    model = build_model()
    assert model.predict(np.zeros((2, 28, 28, 1), dtype="float32"), verbose=0).shape == (2, 10)


def test_model_is_compiled():
    assert build_model().optimizer is not None
