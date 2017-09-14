import tensorflow as tf

x1 = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x1")
x2 = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x2")
y1 = x1 + x2
y2 = x1 * x2

builder = tf.saved_model.builder.SavedModelBuilder('./model')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	builder.add_meta_graph_and_variables(sess,
		[tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			'add': tf.saved_model.signature_def_utils.predict_signature_def(
				inputs= {"x1": x1, "x2": x2},
				outputs= {"result": y1}),
			'multiply': tf.saved_model.signature_def_utils.predict_signature_def(
				inputs= {"x1": x1, "x2": x2},
				outputs= {"result": y2})})

	builder.save()

	print(sess.run([y1, y2], feed_dict={x1: [[1]], x2: [[0]]}))
