import numpy as np
import dlib
import cv2
from lip_detect.cleaning import *
from icecream import ic
from PIL import Image as im

LIP_MARGIN = 0.4                # Marginal rate for lip-only image.
RESIZE = (70,30)
# VIDEO_PATH = <videopath!!!!!!!!!>
DATFILE =  "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('lip_detect/shape_predictor_68_face_landmarks.dat')

def lip_detect(frame):
    ic('RUNNING')
    ic(frame.shape)
    face_rects = detector(frame,1)             #detects face
    if len(face_rects) < 1:                 #no faces
        print("No face detected")
    if len(face_rects) > 1:                  #too many faces
        print("Too many faces")
    rect = face_rects[0]                    #proper number of faces
    landmark = predictor(frame, rect)   #detect face landmarks
    landmark = shape_to_list(landmark)

    lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
    lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])  # Lip landmark sorted for determining lip region
    lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
    x_add = int((lip_x[-1][0]-lip_x[0][0])*LIP_MARGIN*1)                     # Determine Margins for lip-only image
    y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN*2)
    crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)
    ic(frame.shape)
    cropped = frame[crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]        # Crop image
    ic(cropped.shape)
    cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)       # Resize

    ic(cropped.shape)
    # gray_image_list = []

    # for i, image in enumerate(cropped_img_list):
    #     gray_frame_list = []
    #     for j, frame in enumerate(image):
    #         gray_image = makeitgray(cropped_img_list,i,j)
    #         # comment this out if you don't need the shape (X, X, 1)
    #         gray_image = np.expand_dims(gray_image, axis=2)

    #         gray_frame_list.append(gray_image)
    #         gray_image_list.append(gray_frame_list)

    return cropped


vid = cv2.VideoCapture("/Users/cbeams/code/projects-1561/lip-reader/test.mp4")

success, frame = vid.read()
img = im.fromarray(frame).convert('L')  # Convert image into grayscale
f_g = np.array(img)
ic(f_g.shape)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale

vid.release()
print(lip_detect(f_g))

#testing
if __name__ == "main":
    vid = cv2.VideoCapture("/Users/cbeams/code/projects-1561/lip-reader/test.mp4")

    success, frame = vid.read()
    img = im.fromarray(frame).convert('L')  # Convert image into grayscale
    f_g = np.array(img, dtype='f')
    ic(f_g.shape)
    vid.release()
    print(lip_detect(np.array(img)))
