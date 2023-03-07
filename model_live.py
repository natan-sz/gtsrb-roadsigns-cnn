import os
import cv2
import numpy as np
from PIL import Image
from keras import models
import pickle
from road_labels import label_name
import numpy as np
import argparse

model_dir = "models"
models_avail = "alpha, beta, gamma, alpha_aug, beta_aug, gamma_aug, alpha_aug_fix, beta_aug_fix, gamma_aug_fix"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="models available:"+models_avail ,required=True)
args = parser.parse_args()

# Model selection, i know i could do this cleaner but this is what it is for now ¯\_(ツ)_/¯
if args.model == "alpha_aug":
    model_file = os.path.join(model_dir,"model_alpha_aug.pickle")
elif args.model == "beta_aug":
    model_file = os.path.join(model_dir,"model_beta_aug.pickle")
elif args.model == "gamma_aug":
    model_file = os.path.join(model_dir,"model_gamma_aug.pickle")
elif args.model == "alpha":
    model_file = os.path.join(model_dir,"model_alpha.pickle")
elif args.model == "beta":
    model_file = os.path.join(model_dir,"model_beta.pickle")
elif args.model == "gamma":
    model_file = os.path.join(model_dir,"model_gamma.pickle")
elif args.model == "alpha_aug_fix":
    model_file = os.path.join(model_dir,"model_alpha_aug_fix.pickle")
elif args.model == "beta_aug_fix":
    model_file = os.path.join(model_dir,"model_beta_aug_fix.pickle")
elif args.model == "gamma_aug_fix":
    model_file = os.path.join(model_dir,"model_gamma_aug_fix.pickle")
else:
    print("Model not provided add a --model flag")

# Start video capute and load the saved model
video = cv2.VideoCapture(0)
with open(model_file, 'rb') as f:
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

        print(f"Predicted: {label_name[prediction]}")
        

        out = np.uint8(im)
        cv2.imshow("Capturing", out)
        key=cv2.waitKey(1)

        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
