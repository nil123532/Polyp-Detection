import math

from pykalman import KalmanFilter
import numpy as np
from numpy import ma

class Object():
    def __init__(self, transition_matrix, observation_matrix, initial_state, confidence, smoothing_factor):
        transition_covariance = np.eye(8)
        observation_covariance = np.eye(4) * 0

        #transition_covariance = np.asarray(transition_covariance)
        #observation_covariance = np.asarray(observation_covariance)

        #transition_offsets = [-0.05100077, -0.02864671,  0.00265585,  0.00708265,  0.0281431,  -0.00251459, -0.00030906, -0.00146543]
        #observation_offsets = [-0.0093744,  -0.0099549,   0.00107086,  0.00301022]

        #transition_offsets = np.asarray(transition_offsets)
        #observation_offsets = np.asarray(observation_offsets)

        self.kf = KalmanFilter(transition_matrices=transition_matrix,
                               transition_covariance=transition_covariance,
                               #transition_offsets=transition_offsets,
                               observation_matrices=observation_matrix,
                               observation_covariance=observation_covariance,
                               #observation_offsets=observation_offsets,
                               initial_state_mean=initial_state)

        self.transition_matrix = transition_matrix

        self.mean = np.asarray(initial_state)
        #self.covariance = np.identity(8)
        self.covariance = np.zeros((len(initial_state), len(initial_state)))

        self.masked = 0
        self.length = 0

        self.confidence = confidence
        self.smoothing_factor = smoothing_factor 
        
        #self.classification=0;

    def add_observation(self, observation, masked=False, confidence=0):


        #transition_covariance = np.eye(8)
        #observation_covariance = np.zeros((4,4))

        #print(self.kf.transition_cov)

        (self.mean, self.covariance) = self.kf.filter_update(self.mean, self.covariance, observation)

        #print(self.covariance)

        self.length += 1
        #print(self.length)
        if masked:
            self.masked += 1
        else:
            self.masked = 0

        #self.confidence = (self.smoothing_factor * confidence) + ((1 - self.smoothing_factor) * self.confidence)
        self.confidence = confidence
        #self.classification=class;
        
    def predict(self):
        return np.matmul(self.transition_matrix, self.mean)

class Tracker():
    def __init__(self, min_confidence=0.05,
                 min_new_confidence=0.1,
                 exclusive_threshold=500,
                 match_threshold=500,
                 max_unseen=2,
                 smoothing_factor=0.5,
                 area_weight=0.5):
        t = 1
        self.transition_matrix = np.asarray([[1, 0, 0, 0, t, 0, 0, 0],
                                            [0, 1, 0, 0, 0, t, 0, 0],
                                            [0, 0, 1, 0, 0, 0, t, 0],
                                            [0, 0, 0, 1, 0, 0, 0, t],
                                            [0, 0, 0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 1]])
        self.observation_matrix = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0]])
        #self.initial_state = np.asarray([pos_x, pos_y, width, height, 0, 0, 0, 0])

        self.current_objects = []

        self.min_confidence = min_confidence
        self.min_new_confidence = min_new_confidence
        self.exclusive_threshold = exclusive_threshold
        self.match_threshold = match_threshold
        self.max_unseen = max_unseen
        self.smoothing_factor = smoothing_factor
        self.area_weight = area_weight

    def update(self, bboxes):
        updated_objects = []

        bboxes = bboxes[bboxes[:, 1] >= self.min_confidence]

        if len(bboxes) > 0:
            # remove bounding boxes that are similar and have lower confidence
            bbox_areas_1 = (bboxes[:, 4] - bboxes[:, 2]) * (bboxes[:, 5] - bboxes[:, 3])
            area_scores_1 = np.tile(bbox_areas_1.reshape((len(bboxes), 1)), (1, len(bboxes)))
            area_scores_1 -= bbox_areas_1
            area_scores_1 = np.sqrt(np.abs(area_scores_1))

            bbox_x_locations = (bboxes[:, 2] + bboxes[:, 4]) / 2
            bbox_y_locations = (bboxes[:, 3] + bboxes[:, 5]) / 2

            bbox_locations_orig = np.dstack((bbox_x_locations, bbox_y_locations))
            bbox_locations = bbox_locations_orig.reshape((len(bboxes), 1, 2))
            bbox_locations = np.tile(bbox_locations, (len(bboxes), 1))
            bbox_locations[:, :] -= bbox_locations_orig
            bbox_locations[:, :, :] **= 2

            bbox_distance = np.sum(bbox_locations, axis=2)
            bbox_distance = np.sqrt(bbox_distance)

            scores = area_scores_1 + bbox_distance

            np.fill_diagonal(scores, np.nan)

            remove = []

            for i in range(len(bboxes)):
                for j in range(i + 1):
                    if i != j:
                        if scores[i, j] < self.exclusive_threshold:
                            remove.append(i)

            keep = set(range(len(bboxes))) - set(remove)

            bboxes = bboxes[list(keep)]

            # calculate matching bboxes
            area_scores = np.zeros((len(self.current_objects), len(bboxes)))

            bbox_areas = (bboxes[:, 4] - bboxes[:, 2]) * (bboxes[:, 5] - bboxes[:, 3])

            for i in range(len(self.current_objects)):
                current_area = 4 * self.current_objects[i].predict()[2] * self.current_objects[i].predict()[3]

                for j in range(len(bboxes)):
                    area_scores[i, j] = math.sqrt(abs(current_area - bbox_areas[j]))

            distance_scores = np.zeros((len(self.current_objects), len(bboxes)))

            for i in range(len(self.current_objects)):
                for j in range(len(bboxes)):
                    distance_scores[i, j] = math.sqrt((self.current_objects[i].predict()[0] - ((bboxes[j, 2] + bboxes[j, 4]) / 2)) ** 2 +
                                                      (self.current_objects[i].predict()[1] - ((bboxes[j, 3] + bboxes[j, 5]) / 2)) ** 2)

            total_scores = area_scores + distance_scores

            total_scores_copy = total_scores.copy()

            used_bboxes = []

            if len(self.current_objects) > 0:

                min_score_indices = []

                for i in range(min(len(self.current_objects), len(bboxes))):
                    min_score_indices.append(np.unravel_index(total_scores_copy.argmin(), total_scores_copy.shape))

                    total_scores_copy[min_score_indices[-1][0], :] = np.inf
                    total_scores_copy[:, min_score_indices[-1][1]] = np.inf

                for i in range(len(min_score_indices)):
                    #print(total_scores[min_score_indices[i][0], min_score_indices[i][1]])
                    if total_scores[min_score_indices[i][0], min_score_indices[i][1]] < self.match_threshold:
                        bbox_index = min_score_indices[i][1]

                        self.current_objects[i].add_observation([(bboxes[bbox_index, 2] + bboxes[bbox_index, 4]) / 2,
                                                                 (bboxes[bbox_index, 3] + bboxes[bbox_index, 5]) / 2,
                                                                 (bboxes[bbox_index, 4] - bboxes[bbox_index, 2]) / 2,
                                                                 (bboxes[bbox_index, 5] - bboxes[bbox_index, 3]) / 2],
                                                                 False,
                                                                 bboxes[bbox_index, 1])

                        used_bboxes.append(bbox_index)
                        updated_objects.append(i)
                        #print(str(i) + ": observation")

            for i in range(len(bboxes)):
                if i not in used_bboxes:
                    if bboxes[i][1] > self.min_new_confidence:
                        obj = Object(self.transition_matrix, self.observation_matrix, [(bboxes[i, 2] + bboxes[i, 4]) / 2,
                                                                                       (bboxes[i, 3] + bboxes[i, 5]) / 2,
                                                                                       (bboxes[i, 4] - bboxes[i, 2]) / 2,
                                                                                       (bboxes[i, 5] - bboxes[i, 3]) / 2,
                                                                                        0, 0, 0, 0],
                                                                                       bboxes[i, 1],
                                                                                        self.smoothing_factor)

                        self.current_objects.append(obj)
                        #print("new")

        self.current_objects = [obj for obj in self.current_objects if obj.masked < self.max_unseen]

        for i in range(len(self.current_objects)):
            if i not in updated_objects:
                arr = ma.asarray([[0, 0, 0, 0]], dtype=float)
                arr[0] = ma.masked
                self.current_objects[i].add_observation(arr, masked=True)
                #print(str(i) + ": no observation")

        return self.current_objects

    def reset(self):
        self.current_objects = []
