import numpy
import tensorflow as tf
import os.path
import random
import math

BATCH_SIZE =50
INPUT_SIZE = 256
KERNEL_SIZE = 3



'''
import model
import dataset
import tensorflow as tf
test_examples = dataset.load_examples('/mnt/satellite/data/test', 50)
m = model.Model()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
	m.saver.restore(session, '/home/ubuntu/model_sat')
	l1, l2, l3, m1, m2, o1, o2, o3, os = session.run([m.lowlevel1, m.lowlevel2, m.lowlevel3, m.midlevel1, m.midlevel2, m.output1, m.output2, m.output3, m.outputs], feed_dict=m._prepare_feed_dict(test_examples, is_training=False))

for _ in xrange(100):
	train_examples = dataset.load_examples('/mnt/satellite/data/train', 10000)
	m.train(train_examples, test_examples, '/home/ubuntu/model_sat')

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
	m.saver.restore(session, '/home/ubuntu/model_sat')
	outputs = session.run([m.outputs], feed_dict=m._prepare_feed_dict(test_examples, is_training=False))
outputs = outputs.reshape((50, 256, 256))
for i in xrange(50): Image.fromarray((outputs[i, :, :] * 255).astype(numpy.uint8)).save('/home/ubuntu/examples/{}_output.png'.format(i))
for i in xrange(50): Image.fromarray((test_examples[i][0][:, :, :] * 255).astype(numpy.uint8)).save('/home/ubuntu/examples/{}_input_sat.png'.format(i))
for i in xrange(50): Image.fromarray((test_examples[i][1].reshape(256, 256) * 255).astype(numpy.uint8)).save('/home/ubuntu/examples/{}_target.png'.format(i))

varmap = {}
for var in tf.all_variables():
	if 'Adadelta' not in var.name:
		varmap[var.name] = var
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
	m.saver.restore(session, '/home/ubuntu/model1')
	l1w, l1b, l2w, l2b = session.run([varmap['lowlevel1/weights:0'], varmap['lowlevel1/biases:0'], varmap['lowlevel2/weights:0'], varmap['lowlevel2/biases:0']])
'''

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, activation='relu'):
		with tf.variable_scope(name) as scope:
			kernel = tf.get_variable(
				'weights',
				shape=[KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			h = tf.nn.bias_add(
				tf.nn.conv2d(
					input_var,
					kernel,
					[1, stride, stride, 1],
					padding='SAME'
				),
				biases
			)
			h_normalized = tf.contrib.layers.batch_norm(h, center=True, scale=True, updates_collections=None, is_training=self.is_training)
			if activation == 'relu':
				return tf.nn.relu(h_normalized, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(h_normalized, name=scope.name)

	def __init__(self):
		tf.reset_default_graph()

		# whether we are in training mode
		self.is_training = tf.placeholder(tf.bool)

		# input cell weight/directions
		self.inputs = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 4])

		# gaussian noise applied on the OSM road network
		self.targets = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE*2, INPUT_SIZE*2, 2])


		#self.scaledinputs = tf.image.resize_images(self.inputs, [INPUT_SIZE*2, INPUT_SIZE*2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# low level layers
		self.lowlevel1 = self._conv_layer('lowlevel1', self.inputs, 2, 4, 64)
		#self.lowlevel1 = self._conv_layer('lowlevel1', self.scaledinputs, 2, 4, 64)
		self.lowlevel2 = self._conv_layer('lowlevel2', self.lowlevel1, 1, 64, 128)
		self.lowlevel3 = self._conv_layer('lowlevel3', self.lowlevel2, 2, 128, 128)
		self.lowlevel4 = self._conv_layer('lowlevel4', self.lowlevel3, 1, 128, 256)
		self.lowlevel5 = self._conv_layer('lowlevel5', self.lowlevel4, 2, 256, 256)
		self.lowlevel6 = self._conv_layer('lowlevel6', self.lowlevel5, 1, 256, 256)
		self.lowlevel7 = self._conv_layer('lowlevel7', self.lowlevel6, 1, 256, 256)

		# output layers
		self.output1 = self._conv_layer('output1', self.lowlevel7, 1, 256, 128)
		self.output2 = tf.image.resize_images(self.output1, [INPUT_SIZE // 4, INPUT_SIZE // 4], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		self.output3 = self._conv_layer('output3', self.output2, 1, 128, 64)
		self.output4 = self._conv_layer('output4', self.output3, 1, 64, 64)
		self.output5 = tf.image.resize_images(self.output4, [INPUT_SIZE // 2, INPUT_SIZE // 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		self.output6 = self._conv_layer('output6', self.output5, 1, 64, 64)
		self.output7 = tf.image.resize_images(self.output6, [INPUT_SIZE, INPUT_SIZE], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		self.output8 = self._conv_layer('output7', self.output7, 1, 64, 16)
		self.output8_5 = tf.image.resize_images(self.output8, [INPUT_SIZE*2, INPUT_SIZE*2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		self.output9 = self._conv_layer('output9', self.output8_5, 1, 16, 8)
		self.outputs = self._conv_layer('outputs', self.output9, 1, 8, 2, activation='sigmoid')


		#self.l2_loss = tf.reduce_mean(tf.square(self.targets[96:160, 96:160,:] - self.outputs[96:160, 96:160,:]))

		#self.targets_sub = self.targets[1:255, 1:255, :]
		#self.outputs_sub = self.outputs[1:255, 1:255, :]

		#self.targets_sub = self.targets[96:160, 96:160, :]
		#self.outputs_sub = self.outputs[96:160, 96:160, :]

	
		self.l2_loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
		#self.l2_loss_sub = tf.reduce_mean(tf.square(self.targets_sub - self.outputs_sub))

                # compute regularization loss
                small_size = INPUT_SIZE - 2
                no_offset = self.outputs[:, 1:small_size+1, 1:small_size+1, :]
                offset1 = self.outputs[:, 2:small_size+2, 1:small_size+1, :]
                offset2 = self.outputs[:, 0:small_size, 1:small_size+1, :]
                offset3 = self.outputs[:, 1:small_size+1, 2:small_size+2, :]
                offset4 = self.outputs[:, 1:small_size+1, 0:small_size, :]
                rloss1 = tf.reduce_mean(tf.square(offset1 - no_offset))
                rloss2 = tf.reduce_mean(tf.square(offset2 - no_offset))
                rloss3 = tf.reduce_mean(tf.square(offset3 - no_offset))
                rloss4 = tf.reduce_mean(tf.square(offset4 - no_offset))
                self.regularization_loss = (rloss1 + rloss2 + rloss3 + rloss4)*0.05


		

		self.loss = self.l2_loss + self.regularization_loss #+ self.l2_loss_sub

		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)



		#self.check = tf.add_check_numerics_ops()

		#self.loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
		#self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver()

	def _prepare_feed_dict(self, examples, is_training=False):
		return {
			self.inputs: [example[0] for example in examples],
			self.targets: [example[1] for example in examples],
			self.is_training: is_training,
		}

	def _train(self, session, train_examples, test_examples):
		for i in xrange(0, len(train_examples), BATCH_SIZE):
			if i % 1000 == 0:
				print('{}/{}'.format(i, len(train_examples)))
			batch = train_examples[i:i+BATCH_SIZE]
			session.run(self.optimizer, feed_dict=self._prepare_feed_dict(batch, is_training=True))
			

		train_loss_reg = session.run(self.regularization_loss, feed_dict=self._prepare_feed_dict(train_examples[0:BATCH_SIZE], is_training=False))
		test_loss_reg = session.run(self.regularization_loss, feed_dict=self._prepare_feed_dict(test_examples, is_training=False))
		train_loss = session.run(self.loss, feed_dict=self._prepare_feed_dict(train_examples[0:BATCH_SIZE], is_training=False))
		test_loss = session.run(self.loss, feed_dict=self._prepare_feed_dict(test_examples, is_training=False))
		print('Sum train_loss={}, test_loss={}'.format(train_loss, test_loss))
		print('L2  train_loss={}, test_loss={}'.format(train_loss - train_loss_reg, test_loss - test_loss_reg))
		print('Reg train_loss={}, test_loss={}'.format(train_loss_reg, test_loss_reg))
		#with open("/home/ubuntu/songtao/loss.txt","a") as fout:
		#	fout.write("total loss: "+str(train_loss)+","+str(test_loss)+", L2: "+str(train_loss-train_loss_reg)+","+str(test_loss-test_loss_reg)+" reg:"+str(train_loss_reg)+","+str(test_loss_reg)+"\n")
		return train_loss, test_loss

	def train(self, train_examples, test_examples, save_path=None):
		loss_train = 0
		loss_test = 0
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
			if save_path is not None and os.path.isfile(save_path + '.meta'):
				print('debug: loading from {}'.format(save_path))
				self.saver.restore(session, save_path)
			else:
				print('debug: re-initializing')
				session.run(self.init_op)

			loss_train, loss_test = self._train(session, train_examples, test_examples)

			if save_path is not None:
				self.saver.save(session, save_path)
				#if loss_train < 0.026:
				#	self.saver.save(session, save_path+'X_'+str(loss_train))
		return loss_train, loss_test
			

'''	def apply(self, examples, save_path):
		converted_examples = self._convert_examples(examples)
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
			self.saver.restore(session, save_path)
			output, classify_loss, colorization_loss, loss = session.run(
				[self.output, self.classify_loss, self.colorization_loss, self.loss],
				feed_dict=self._prepare_feed_dict(converted_examples)
			)
			print('classify_loss={}, colorization_loss={}, combined loss={}'.format(classify_loss, colorization_loss, loss))
			return output

		images = []
		for i in xrange(len(examples)):
			img_cielab = examples[i][0]
			#output_big = (skimage.transform.pyramid_expand(output[i], upscale=2) * 256.0) - 128.0
			image = numpy.concatenate([skimage.transform.downscale_local_mean(img_cielab[:, :, 0:1], (2, 2, 1)), ((output[i] * 256.0) - 128.0).astype('uint8')], axis=2)
			images.append(image)
		return images

	def test(self, examples, save_path):
		converted_examples = self._convert_examples(examples)
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as session:
			self.saver.restore(session, save_path)
			losses = []
			for i in xrange(0, len(converted_examples), BATCH_SIZE):
				batch = converted_examples[i:i+BATCH_SIZE]
				loss = session.run(self.colorization_loss, feed_dict=self._prepare_feed_dict(batch, is_training=False))
				losses.append(loss)
			return numpy.mean(losses)'''
