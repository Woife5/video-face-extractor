## Set the filename for the next image to be written
counter = 0

## Is this is set to true, existing images in /faces will be overwritten, otherwise
## the existing image names will be skipped
overwrite_images = False

## Set set all videos you want to extract faces from
## These videos should be located in the ./srcvideos directory
input_file_names = [ "movie.mov" ]

import os

## If you want to extract the faces from all videos uncomment the following lines
# def is_video_file(file):
#     video_endings = [".mov", ".mp4"]
#     for ending in video_endings:
#         if file.endswith(ending):
#             return True
#     return False

# input_file_names = list(filter(is_video_file, os.listdir("./srcvideos")))

import face_recognition
import cv2
import numpy as np
import math
from PIL import Image

errorcount = 0

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

for i in input_file_names:

    # Open the input movie file
    input_movie = cv2.VideoCapture("srcvideos/" + i)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize some variables
    face_locations = []
    frame_number = 0

    current_path = os.getcwd()

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Find all the faces in the current frame of video
        face_landmarks = face_recognition.face_landmarks(frame, model="small")

        # Align and save the face
        for face in face_landmarks:
            right_eye = tuple([(face["right_eye"][0][0] + face["right_eye"][1][0]), face["right_eye"][0][1] + face["right_eye"][1][1]])
            right_eye = tuple([(int)(right_eye[0]/2), (int)(right_eye[1]/2)])

            left_eye = tuple([(face["left_eye"][0][0] + face["left_eye"][1][0]), face["left_eye"][0][1] + face["left_eye"][1][1]])
            left_eye = tuple([(int)(left_eye[0]/2), (int)(left_eye[1]/2)])

            if left_eye[1] > right_eye[1]:
                point_3rd = (right_eye[0], left_eye[1])
                direction = -1 #rotate same direction to clock
            else:
                point_3rd = (left_eye[0], right_eye[1])
                direction = 1 #rotate inverse direction of clock

            a = euclidean_distance(left_eye, point_3rd)
            b = euclidean_distance(right_eye, left_eye)
            c = euclidean_distance(right_eye, point_3rd)
            cos_a = (b*b + c*c - a*a)/(2*b*c)

            angle = np.arccos(cos_a)
            angle = (angle * 180) / math.pi

            if direction == -1:
                angle = 90 - angle

            percent = 1 / (b / 40)

            new_left_eye = tuple(i*percent for i in left_eye)
            new_right_eye = tuple(i*percent for i in right_eye)

            width = int(frame.shape[1] * percent)
            height = int(frame.shape[0] * percent)
            dim = (width, height)

            new_img = cv2.resize(frame, dim)

            new_img = Image.fromarray(new_img)
            new_img = np.array(new_img.rotate(direction * angle))

            center_x=(int)((new_right_eye[0] + new_left_eye[0]) / 2)
            center_y=(int)((new_right_eye[1] + new_left_eye[1]) / 2)

            new_img = new_img[center_y-30:center_y+70, center_x-50:center_x+50]

            if(len(new_img) < 100 or len(new_img[0]) < 100):
                print("File {} at frame {}: Image too small. Skipping...".format(i, frame_number))
                errorcount += 1
            else:
                try:
                    file_path = "{}/faces/{:0>6d}.png".format(current_path, counter)

                    # Check if the file already exists; this can be used to fill deleted files
                    # Comment out the entire loop to just overwrite images
                    while os.path.isfile(file_path) and not overwrite_images:
                        counter += 1
                        file_path = "{}/faces/{:0>6d}.png".format(current_path, counter)
                    
                    cv2.imwrite(file_path, new_img)
                    print("Wrote image {:0>6d}.png".format(counter))
                    counter += 1
                except cv2.error:
                    print("An error occured in file {} at frame {}. Skipping...".format(i, frame_number))
                    errorcount += 1
            
        # Write the resulting image to the output video file
        print("Processed Frame {} / {}".format(frame_number, length))
        
    input_movie.release()
    cv2.destroyAllWindows()

print("Last written Image: {:0>6d}.png".format(counter-1))
print("Images Skipped due to errors: {}".format(errorcount))
# All done!
