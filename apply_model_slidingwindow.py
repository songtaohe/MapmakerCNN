import numpy
import model
import tensorflow as tf
from PIL import Image

import scipy.ndimage
import scipy.misc

model.BATCH_SIZE = 1

m = model.Model()

satallite = scipy.ndimage.imread("example/googlemap.png").astype(numpy.uint8)
gps = scipy.ndimage.imread("example/gps.png").astype(numpy.uint8)

conv_input = numpy.zeros((256,256,4))
conv_mask = numpy.zeros((512,512,2))

xsize = numpy.shape(satallite)[0]
ysize = numpy.shape(satallite)[1]


finalresult = numpy.zeros((xsize*2, ysize*2, 3))

xoffset = 128
yoffset = 128

sub_output_size = 64


with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
	m.saver.restore(session, 'model/modelX_0.0232608')
	while True:
		yoffset = 128
		while True:
			conv_input[:,:,0] = gps[xoffset-128:xoffset+128,yoffset-128:yoffset+128]
			conv_input[:,:,1] = satallite[xoffset-128:xoffset+128,yoffset-128:yoffset+128,0]
			conv_input[:,:,2] = satallite[xoffset-128:xoffset+128,yoffset-128:yoffset+128,1]
			conv_input[:,:,3] = satallite[xoffset-128:xoffset+128,yoffset-128:yoffset+128,2]

			conv_input_ = conv_input.astype(numpy.float64) / 255.0

			examples = []

			examples.append((conv_input_, conv_mask))
			
			outputs = session.run([m.outputs], feed_dict=m._prepare_feed_dict(examples, is_training=False))[0]
								
			outputs = outputs.reshape((512, 512,2))

			finalresult[xoffset*2-sub_output_size:xoffset*2+sub_output_size, yoffset*2-sub_output_size:yoffset*2+sub_output_size, 1] = outputs[256-sub_output_size: 256+sub_output_size, 256-sub_output_size:256+sub_output_size,0]
			finalresult[xoffset*2-sub_output_size:xoffset*2+sub_output_size, yoffset*2-sub_output_size:yoffset*2+sub_output_size, 2] = outputs[256-sub_output_size: 256+sub_output_size, 256-sub_output_size:256+sub_output_size,1]

	
			yoffset = yoffset + sub_output_size

			if yoffset + 128 > numpy.shape(gps)[1]:
				break

		xoffset = xoffset + sub_output_size
		if xoffset + 128 > numpy.shape(gps)[0]:
			break

		print(xoffset)


Image.fromarray((finalresult * 255).astype(numpy.uint8)).save("output.png")

