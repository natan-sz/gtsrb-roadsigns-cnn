import cv2
import numpy as np
from PIL import Image
from keras import models
import pickle
from road_labels import label_name
import numpy as np

# Start video capute and load the saved model
video = cv2.VideoCapture(0)
with open('model_alpha_aug.pickle', 'rb') as f:
    model = pickle.load(f)

while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')
        width, height = im.size

        left = (width/2)-(height/2)
        top = 0
        right = (width/2)+(height/2)
        bottom = height

        # Crop out a square
        im = im.crop((left, top, right, bottom))
        im_resized = im.resize((30,30))

        img_array = np.array(im_resized)

        img_array = np.expand_dims(img_array, axis=0)

        probabilities = model.predict(img_array)
        prediction = np.argmax(probabilities)

        print(f"{label_name[prediction]} ({probabilities[0][prediction]}%)")
        

        out = np.uint8(im)
        cv2.imshow("Capturing", out)
        key=cv2.waitKey(1)

        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
