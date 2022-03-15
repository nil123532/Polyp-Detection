import numpy as np
from tensorflow.keras.models import load_model

class BBPS_Scorer:
    def __init__(self, model_file, smoothing_factor=0.2):
        self.model = load_model(model_file)
        self.score = None
        self.smoothing_factor = smoothing_factor

        self.model.predict(np.ones((1, 128, 128, 3)))

    def predict(self, image):
        score = self.model.predict(image)[0][0]

        if self.score is None:
            self.score = score
        else:
            self.score = self.smoothing_factor * score + (1 - self.smoothing_factor) * self.score

        return self.score
