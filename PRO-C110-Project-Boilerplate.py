# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		#resize the frame
		resizedFrame = cv2.resize(frame, (244, 244))
		
		# expand the dimensions
		converting = np.array(resizedFrame, dtype = np.float32)
        expand = np.expand_dims(converting, axis = 0)
		
		# normalize it before feeding to the model

		normalizedFrame= expand/255.0
		
		# get predictions from the model
		
		prediction = model.predict(normalizedFrame)
        print("prediction", prediction)
		
		# displaying the frames captured
		
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
