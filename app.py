from flask import Flask, render_template, request
from keras.models import load_model
import keras
from keras.utils import load_img, img_to_array
import cv2
from PIL import Image
import numpy as np
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
def resize(img):
	image=Image.open(img)
	image = np.array(image)
	reshaped_image = np.reshape(image, (28, 28, 1))/255.0
	
	# reshaped_image = Image.fromarray(reshaped_image)
	return reshaped_image
	

app = Flask(__name__)

my_dictionary = {
    0: 'T-shirt/top',
    1: 'trouser',
    2: ' pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'Ankel boots'
}
model = load_model('model3.h5')

model.make_predict_function()
def pre(image_path):
	# img =load_img(image_path)
	y=f'{image_path}'
	x=resize(y)
	# original_image = cv2.imread(img)
	# grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	
	# image=np.array(img) / 255.0
	# i = np.reshape(image,(28, 28))
	# x = image.img_to_array(img)
	prob=model.predict(x)
	class_idx=np.argmax(prob)
	
	return my_dictionary[class_idx]
	# prob = model.predict(x)  # Get class probabilities
    # class_idx = np.argmax(prob)  # Get the index of the class with the highest probability
# def predict_label(img_path):
#     i = image.load_img(img_path)
#     i = image.img_to_array(i) / 255.0
#     i = i.reshape( None,28, 28)
#     prob = model.predict(i)  # Get class probabilities
#     class_idx = np.argmax(prob)  # Get the index of the class with the highest probability
#     return my_dictionary[class_idx]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = pre(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)