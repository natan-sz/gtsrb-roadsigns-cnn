# GTSRB Roadsigns (CNN)
Convolutional Neural Network solution for the GTSRB Roadsigns dataset.

<img width=400 src="https://production-media.paperswithcode.com/datasets/GTSRB-0000000633-9ce3c5f6_Dki5Rsf.jpg"/>

### Quickstart

To install dependencies, run the below

	$ pip3 install -r requirements.txt

After this you should be able to run the notebook no problem. The models have been setup to load in the models from pickle files if they exist.

### OpenCV model implementation
To run the opencv webcam implementation run the following with the below flag.
Currently this works but the predicitons are completely wrong most of the time. Requires more time to fix.
	# model_names = alpha_aug, beta_aug, gamma_aug
	$ python3 model_live.py -m <model_name>
