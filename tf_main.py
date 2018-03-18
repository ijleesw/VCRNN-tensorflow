"""
Implemetation of MNIST classification with RNN in tensorflow based on following urls:

https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23#file-tf_lstm-py
https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
http://www.osia.or.kr/board/include/download.php?no=63&db=data2&fileno=4
"""

import tensorflow as tf
import tf_VCRNN as tf_vcrnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


########## GRAPH DEFINITION ##########

INPUT_DIM     = 28      # IMPUT_DIM pixels per timesteps
HIDDEN_DIM    = 98
OUTPUT_DIM    = 10       
LEARNING_RATE = 0.001
TIME_STEPS    = 28		# IMPUT_DIM * TIME_STEPS = 784
# NUM_LAYERS    = 1


X = tf.placeholder(tf.float32, (None, TIME_STEPS, INPUT_DIM))   # (batch_size, time, in)
y = tf.placeholder(tf.float32, (None, OUTPUT_DIM))              # (batch_size, out)
X_input = tf.unstack(X, TIME_STEPS, 1)


# cell = tf_vcrnn.VCRNNCell(HIDDEN_DIM, sharpness=1.0, reuse=tf.AUTO_REUSE)
cell = tf_vcrnn.VCGRUCell(HIDDEN_DIM, sharpness=1.0, reuse=tf.AUTO_REUSE)
# cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_DIM, reuse=tf.AUTO_REUSE)
# cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_DIM, reuse=tf.AUTO_REUSE)      # Choose one

# cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.5)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NUM_LAYERS) 


rnn_outputs, rnn_states = tf.nn.static_rnn(cell, X_input, dtype="float32")

final_W = tf.Variable(tf.random_normal([HIDDEN_DIM, 10]))
final_b = tf.Variable(tf.random_normal([10]))
predicted_y = tf.matmul(rnn_outputs[-1], final_W) + final_b

# compute cross entropy
loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits=predicted_y, labels=y))

# optimize with adam
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
accuracy = tf.reduce_mean(
		tf.cast(tf.equal(tf.argmax(predicted_y, 1), tf.argmax(y, 1)), tf.float32))



########## TRAINING ##########

BATCH_SIZE = 128

X_val, y_val = mnist.train.next_batch(batch_size=BATCH_SIZE*5)
X_val = X_val.reshape((BATCH_SIZE*5, TIME_STEPS, INPUT_DIM))

sess= tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(1000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=BATCH_SIZE)
    X_batch = X_batch.reshape((BATCH_SIZE, TIME_STEPS, INPUT_DIM))
    
    epoch_error = sess.run([loss, optimizer], feed_dict={X: X_batch, y: y_batch})[0]
    
    if epoch==0 or epoch%20 == 19:
        valid_accuracy = sess.run(accuracy, feed_dict={X: X_val, y: y_val})
        print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%"
        			% (epoch+1, epoch_error, valid_accuracy * 100.0))
