import tensorflow as tf
xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random_uniform([1], -100, 100)) #-100부터 100까지랜덤한변수
b = tf.Variable(tf.random_uniform([1], -100, 100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X+b
cost = tf.reduce_mean(tf.square(H - Y))  #거리구해서 평균값구하기
a = tf.Variable(0.01)  #하강할때 얼마나할것인가
optimizer = tf.train.GradientDescentOptimizer(a) #경사하강라이브러리
train = optimizer.minimize(cost)
init = tf.global_variables_initializer() 
sess = tf.Session()
sess.run(init)
for i in range(5001):  #인덱스 5000까지반복
    sess.run(train, feed_dict = {X: xData, Y: yData}) #학습진행
    if i %500 == 0:  #500번에 한번씩 과정출력
        print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
print (sess.run(H, feed_dict={X: [8]}))#모든학습후 원하는 결과출력
