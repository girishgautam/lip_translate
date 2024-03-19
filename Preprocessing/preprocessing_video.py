import numpy as np
import dlib
import cv2
import os
from py_files.cleaning import *


LIP_MARGIN = 0.4                # Marginal rate for lip-only image.
RESIZE = (70,30)
# VIDEO_PATH = <videopath!!!!!!!!!>
DATFILE =  "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/mathildaweston/code/girishgautam/lip_translate/py_files/shape_predictor_68_face_landmarks.dat')

def final_vids(VIDEO_PATH):
    cropped_img_list=[] # Iterate on video files

    if VIDEO_PATH.endswith('.mpg'):
            #vid_path = VIDEO_PATH + vid_name
            vid = cv2.VideoCapture(VIDEO_PATH)

            frames = []               # A list to hold frame images
            frames_colour = []         # A list to hold original frame images
            while(True):
                success, frame = vid.read()
                    # Read frame
                if not success:
                    break                           # Break if no frame to read left
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
                frames.append(gray)                  # Add image to the frame buffer
                frames_colour.append(frame)

            vid.release()

            landmarks = []
            for (i, image) in enumerate(frames):          #iterate on frame lis
                face_rects = detector(image,1)             #detects face
                if len(face_rects) < 1:                 #no faces
                    print(f"No face detected: {VIDEO_PATH}")
                    continue
                if len(face_rects) > 1:                  #too many faces
                    print(f"Too many faces: {VIDEO_PATH}")
                    continue
                rect = face_rects[0]                    #proper number of faces
                landmark = predictor(image, rect)   #detect face landmarks
                landmark = shape_to_list(landmark)
                landmarks.append(landmark)

            cropped_img = []
            for (i,landmark) in enumerate(landmarks):
                lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
                lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
                lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
                x_add = int((lip_x[-1][0]-lip_x[0][0])*LIP_MARGIN*1)                     # Determine Margins for lip-only image
                y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN*2)
                crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)
                cropped = frames_colour[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]        # Crop image
                cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)       # Resize
                cropped_img.append(cropped)


            cropped_img_list.append(cropped_img)

    gray_image_list = []

    for i, image in enumerate(cropped_img_list):
        gray_frame_list = []
        for j, frame in enumerate(image):
            gray_image = makeitgray(cropped_img_list,i,j)
            # comment this out if you don't need the shape (X, X, 1)
            gray_image = np.expand_dims(gray_image, axis=2)

            gray_frame_list.append(gray_image)
            gray_image_list.append(gray_frame_list)

    standardized_list=[]

    for vid in gray_image_list:
        #print(type(vid))
        vid= np.array(vid)
        standard_vid = standardize(vid)
        standardized_list.append(standard_vid)


    processed_vids= standardized_list
    print(len(processed_vids[0]),len(processed_vids[0][0]),len(processed_vids[0][0][0]),len(processed_vids[0][0][0][0]))
    return processed_vids

#testing
final_vids("/home/mathildaweston/code/girishgautam/lip_translate/raw_data/test_mathilda_mpg")
