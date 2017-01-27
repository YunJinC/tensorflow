# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder("float", [None, 784]) #이미지 입력 None은 이미지 개수, 784는 28*28
W = tf.Variable(tf.zeros([784,10])) #W(weight)값 784는 28*28 10은 0~9숫자
b = tf.Variable(tf.zeros([10])) #b(bias)값 0~9t숫자
matm=tf.matmul(x,W)
y = tf.nn.softmax(tf.matmul(x,W) + b) #softmax함수. 인자는 tensor
y_ = tf.placeholder("float", [None,10]) #cost함수 cross entropy error를 위한 변수
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #cross entropy error 함수
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#cross entropy error를 줄이면서 훈련
sess = tf.Session()
sess.run(tf.initialize_all_variables())#변수 초기화 및 세션 실행
for i in range(1000):
 batch_xs, batch_ys = mnist.train.next_batch(100) #무작위 데이터 100 추출
 sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #무작위 데이터 주입
 correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #예측값과 실제 레이블이 같은지 비교
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 확률로 변경
 print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})