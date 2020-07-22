import utils
import collections
import facial_detection

import cv2
from keras.models import load_model

name = input('What is your name?')

model = load_model('siamese_nn.h5', custom_objects={'contrastive_loss': utils.contrastive_loss, 
                    'euclidean_distance': utils.euclidean_distance})

# preprocessing user image
true_img = cv2.imread('true_img.png', 0)
true_img = true_img.astype('float32')/255
true_img = cv2.resize(true_img, (92, 112))
true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1)

video_capture = cv2.VideoCapture(0)
preds = collections.deque(maxlen=15)

while True:
    # capture frame-by-frame
    _, frame = video_capture.read()

    # detect faces
    frame, face_img, face_coords = facial_detection.detect_faces(frame, draw_box=False)

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32')/255
        face_img = cv2.resize(face_img, (92, 112))
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
        
        preds.append(model.predict([true_img, face_img])[0][0])
        x,y,w,h = face_coords
        if len(preds) == 15 and sum(preds)/15 >= 0.3:
            text = 'Identity: {}'.format(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5) 
        elif len(preds) < 15:
            text = "Identifying ..."
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5) 
        else:
            text = "Identity Unknown!"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = utils.write_on_frame(frame, text, face_coords[0], 
                                     face_coords[1]-10)
    else:
        # clear existing predictions if no face detected 
        preds = collections.deque(maxlen=15) 

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()