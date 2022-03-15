import numpy as np
from tensorflow.keras.models import load_model

class Cecum_Detector:
    def __init__(self, model_file, smoothing_factor=0.2, threshold=0.5, cecum_reached_count_threshold=10):
        self.model = load_model(model_file)
        self.score = None
        self.smoothing_factor = smoothing_factor
        self.threshold = threshold
        self.cecum_reached = False
        self.cecum_reached_count = 0
        self.cecum_reached_count_threshold = cecum_reached_count_threshold

        self.model.predict(np.ones((1, 128, 128, 3)))

    def predict(self, image):
        if self.cecum_reached:
            return True
        else:
            score = self.model.predict(image)[0][0]

            if self.score is None:
                self.score = score
            else:
                self.score = self.smoothing_factor * score + (1 - self.smoothing_factor) * self.score

        if self.score < self.threshold:
            self.cecum_reached_count += 1
        else:
            self.cecum_reached_count = 0

        if self.cecum_reached_count > self.cecum_reached_count_threshold:
            self.cecum_reached = True

        return self.cecum_reached
