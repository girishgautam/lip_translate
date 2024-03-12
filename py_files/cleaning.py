import numpy as np

def shape_to_list(shape):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def makeitgray(image_pls, i, j):
    return np.dot(image_pls[i][j][...,:3], [0.2989, 0.5780, 0.1440])

def standardize(vid):
    mean_vid = vid.mean(axis=0)
    std_vid = vid.std(axis=0)
    return (vid-mean_vid)/std_vid
