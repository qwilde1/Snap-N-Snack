import numpy as np
import os
import six.moves.urllib as urllib
import sys
import socket
import json
#import tarfile
import tensorflow as tf
#import zipfile
import socketserver

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from utils import label_map_util

from utils import visualization_utils as vis_util

HOST = requests.get('https://api.ipify.org').text
PORT = 8090

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

def center_coord(max_width,max_height,box):#box is ymin, xmin, ymax, xmax
	#(left, right, top, bottom) = (box[1] * max_width, box[3] * max_width, box[0] * max_height, box[2] * max_height)
	#center_x = (left + right) / 2
	#center_y = (top + bottom) / 2
	center_y = (box[0] + box[2]) / 2
	center_x = (box[1] + box[3]) / 2
	return [center_x,center_y]

#function prepares a json with format 'foodname,center of box coordinate(x,y),certainty percentage'
def prepare_json(mapping,im_width,im_height,boxes,scores,classes,max_elements=10,min_assurance=.70):
	tmp_element_array = []
	count = 0
	for i in range(len(scores)):
		if scores[i] < min_assurance:
			break
		temp_dict = {}
		temp_dict['FoodItem'] = mapping[classes[i]]
		temp_dict['Coordinates'] = center_coord(im_width,im_height,boxes[i])
		temp_dict['Score'] = np.asscalar(scores[i])
		tmp_element_array.append(temp_dict)
		count = count + 1
		if count > 10:
			break
	final_dict = {}
	final_dict['NumItems'] = count
	final_dict['Items'] = tmp_element_array
	json_data = json.dumps(final_dict)
	print(json_data)
	return json_data.encode('utf-8')

	

# Numpy helper code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)

    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

    # Size, in inches, of the output images.

    #url = 'https://firebasestorage.googleapis.com/v0/b/prototype-f8b4d.appspot.com/o/SnapNSnack%2Ff28ce575-fefa-4a50-ac70-789d3cbf274d.jpg?alt=media&token=aba0d94c-0e0a-447f-9ac3-e4bffdf79949'

    # Algorithm to create Tensors, detect the objects, and output an image file with mappings.


	
def analyze_and_return(url):
	IMAGE_SIZE = (12, 8)
	category_index_2 = label_map_util.get_label_map_dict(PATH_TO_LABELS)
	fixed_category_index = {v: k for k,v in category_index_2.items()}
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Don't want a for-loop in final version
			response = requests.get(url)
			image = Image.open(BytesIO(response.content)) #change later so raw image file, rather than file path
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
			# Visualization of the results of a detection.
			# vis_util.visualize_boxes_and_labels_on_image_array(
				# image_np,
				# np.squeeze(boxes),
				# np.squeeze(classes).astype(np.int32),
				# np.squeeze(scores),
				# category_index,
				# use_normalized_coordinates=True,
				# line_thickness=8)
				
			(im_width, im_height) = image.size
			#vis_util.save_image_array_as_png(image_np,sys.argv[1])
			#plt.figure(figsize=IMAGE_SIZE)
			#plt.imsave(imgout,image_np)
			#print(np.squeeze(boxes))
			#print(boxes)
			#print(np.squeeze(classes).astype(np.int32))
			return prepare_json(fixed_category_index,im_width,im_height,np.squeeze(boxes),np.squeeze(scores),np.squeeze(classes).astype(np.int32))

class MyTCPHandler(socketserver.StreamRequestHandler):
	
	def handle(self):
		self.data = self.request.recv(1024).decode('utf-8')
		print('Url Received from phone, processing...')
		reply = analyze_and_return(self.data)
		self.request.sendall(reply)
		print('Json sent, terminating connection')
		self.request.close()
#sys.path.append("..")

if __name__ == "__main__":
	MODEL_NAME = sys.argv[1]
	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box.
	#PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
	PATH_TO_LABELS = 'object-detection.pbtxt'

	# Modify this as you add classes
	#NUM_CLASSES = 2

	#in order to make server modular, use dynamic way to find num_classes
	with open(PATH_TO_LABELS,'r') as content_file:
		content = content_file.read()
	NUM_CLASSES = content.count('item{')


	# Loads the frozen model into memory
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	# Loads the label map
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	#create server and run indefinitely
	server = socketserver.TCPServer((HOST,PORT),MyTCPHandler)
	print('server created, ip: {} port: {}'.format(HOST,PORT))
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		sys.exit(0)

	



		
		
		




# socket.setdefaulttimeout(60)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((socket.gethostname(), 8090))
# s.listen(5)

# while True:
	# connection = None
	# try:
		# print('waiting for connection...')
		# connection, address = s.accept()
		# print('connected!')
		# received_msg = connection.recv(1024)
		# if not received_msg:
			# break
		# print('Url received!')
		# connection.sendall(analyze_and_return(received_msg.decode('utf-8')))
		# connection.close()
		# print('Json sent')
	# except KeyboardInterrupt:
		# if connection:
			# connection.close()
		# break
# s.close()
# print('socket closed, terminating program')
