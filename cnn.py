# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])#이미지 입력 None은 이미지 개수, 784는 28*28
y_ = tf.placeholder("float", shape=[None, 10])#b(bias)값 0~9숫자

x_image = tf.reshape(x, [-1,28,28,1])#이미지 크기 재구성 28x28이미지에 1은 흑백이미지
print "x_image="
print x_image

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#가중치 W를 난수값(random noise)으로 초기화
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)#b(bias)작은 양수를 갖도록 초기화
  return tf.Variable(initial)

def conv2d(x, W):#convolution을 위한 함수
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):#max-pooling을 위함 함수
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


#######1단계 convolution layer#######
W_conv1 = weight_variable([5, 5, 1, 32])#5x5필터에서 쓰일 가중치 W 행렬 32개
b_conv1 = bias_variable([32])#32개 가중치행렬에 대한 bias



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#입력 이미지 x_image 에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv1 을 리턴하고 b_conv1를 더함
h_pool1 = max_pool_2x2(h_conv1)#출력값을 위한 max-pooling
#결과 값으로 12x12x32 feature maps가 나옴

#######2단계 convolution layer#######

W_conv2 = weight_variable([5, 5, 32, 64])#5x5필터에서 쓰일 가중치 W행렬 64개, 32는 1단계 convolution의 출력갑
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#1단계 결과 h_pool1에 대해 convolution을 적용하고 결과를 텐서 W_conv2에 리턴하고 b_conv2를 더함
h_pool2 = max_pool_2x2(h_conv2)#출력값을 위한 max-pooling
#결과값으로 7x7x64 feature maps가 나옴

#######fully connected layer#######
W_fc1 = weight_variable([7 * 7 * 64, 1024])#fully connected layer의 weight 1024개의 뉴런을 사용
b_fc1 = bias_variable([1024])#fully connected layer의 bias

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#softmax에 넣기 위해 직렬화하기 위한 텐서
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#softmax에 넣기 위해 직렬화

#######drop out#######
keep_prob = tf.placeholder("float")#drop out 하지 않을 확률
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#drop out


W_fc2 = weight_variable([1024, 10])#softmax weight
b_fc2 = bias_variable([10])#softmax bias

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#softmax함수

#######training!!!#######
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))#cross entropy error 함수
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#cross entropy error를 줄이면서 훈련
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#예측값과 실제 레이블이 같은지 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))# 확률로 변경

sess = tf.Session()

sess.run(tf.initialize_all_variables())#변수 초기화 및 세션 실행
for i in range(300):
  batch = mnist.train.next_batch(50)
  if i%10 == 0:
    train_accuracy = sess.run( accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))