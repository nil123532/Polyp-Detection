import numpy as np
import os
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

observations = []

for i in range(294, 587):
    annotation = f"/home/david/PaddleDetection/dataset/c4/annotations/{i}.xml"

    if os.stat(annotation).st_size == 0:
        root = None
    else:
        tree = ET.parse(annotation)
        root = tree.getroot()

    if root is not None:
        for child in root:
            if child.tag == "object":
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0

                for position in child[-1]:
                    if position.tag == "xmin":
                        xmin = int(position.text)

                    if position.tag == "xmax":
                        xmax = int(position.text)

                    if position.tag == "ymin":
                        ymin = int(position.text)

                    if position.tag == "ymax":
                        ymax = int(position.text)

                observations.append([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (xmax - xmin) / 2.0, (ymax - ymin) / 2.0, 0, 0, 0, 0])

observations = np.asarray(observations)

t = 1
transition_matrix = np.asarray([[1, 0, 0, 0, t, 0, 0, 0],
                                    [0, 1, 0, 0, 0, t, 0, 0],
                                    [0, 0, 1, 0, 0, 0, t, 0],
                                    [0, 0, 0, 1, 0, 0, 0, t],
                                    [0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
observation_matrix = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]])

#observation_covariance = np.identity(8) * 0

kf = KalmanFilter(transition_matrices=transition_matrix,
                   observation_matrices=observation_matrix,
                  #observation_covariance=observation_covariance,
                   initial_state_mean=observations[0])

kf = kf.em(observations, n_iter=5, em_vars={'transition_covariance',
                                            'observation_covariance',
                                            'observation_offsets',
                                            'transition_offsets'})

print(1)