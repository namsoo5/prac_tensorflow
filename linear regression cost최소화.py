import tensorflow as tf
import matplotlib.pyplot as plt
X = [ 1, 2, 3]
Y = [ 1, 2, 3]

#W = tf.Variable(tf.random_norma([1]), name='weight')
#X = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)
W = tf.Variable(5.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#learning_rate = 0.1
#gradient = tf.reduce_mean((W*X - Y) *X)
#descent = W - learning_rate * gradient
#update = W.assign(descent)
#위의 코드를 간단히 구현되있는 함수를통해가능
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)


sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(100):
    #sess.run(update, feed_dict={X: x_data, Y: y_data})
    #print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    print(step, sess.run(W))
    sess.run(train)
