from pickletools import optimize
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("data/MNIST", one_hot=True, reshape=False)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y_true = tf.compat.placeholder(tf.float32, [None, 10])

filter_1 = 16
filter_2 = 32

weight_1 = tf.Variable(tf.compat.v1.truncated_normal(
    [5, 5, 1, filter_1], stddev=0.1))
bias_1 = tf.Variable(tf.constant(0.1, shape=filter_1))

weight_2 = tf.Variable(tf.compat.v1.truncated_normal(
    [5, 5, filter_1, filter_2], stddev=0.1))
bias_2 = tf.Variable(tf.constant(0.1, shape=filter_2))

weight_3 = tf.Variable(tf.compat.v1.truncated_normal(
    [7*7*filter_2, 256], stddev=0.1))
bias_3 = tf.Variable(tf.constant(0.1, shape=256))

weight_4 = tf.Variable(tf.compat.v1.truncated_normal([256, 10], stddev=0.1))
bias_4 = tf.Variable(tf.constant(0.1, shape=10))

# input [28,28,1]                         strides[batch,x,y,depth] genelde ilk ve son 1 olur
y1 = tf.nn.relu(tf.nn.conv2d(x, weight_1, strides=[
                1, 1, 1, 1], padding="SAME")+bias_1)  # output=[28,28,16]
# maxpooling
y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding="SAME")  # output=[14,14,16]

y2 = tf.nn.relu(tf.nn.conv2d(y1, weight_2, strides=[
                1, 1, 1, 1], padding="SAME")+bias_2)  # output[14,14,32]

# maxpooling
y2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding="SAME")  # output=[7,7,32]
# yeniden boyutlandırma
flattened = tf.reshape(y2, shape=[-1, 7*7*32])

y3 = tf.nn.relu(tf.matmul(flattened, weight_3)+bias_3)
logits = tf.matmul(y3, weight_4)+bias_4
y4 = tf.nn.softmax(logits)

# loss
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# 5e-4=0.0005,1e-4=0.0001,1e-3=0.001,1e4=10000,5e4=50000
optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001).minimize(loss=loss)


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
batch_size = 128
loss_graph = []


def training_step(iterations):
    for i in range(iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimize, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print("Iterations: ", i, " Train accuary: ",
                  train_acc, " Train loss: ", train_loss)


feed_dict_test = {x: mnist.test.images,y_true: mnist.test.labels}

def test_accuary():
    
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Testing accuracy:", acc)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

training_step(2000)
test_accuary()
plot_example_errors()

plt.plot(loss_graph, "k--")
plt.title("loss grafiği")
plt.xlabel("iterasyonlar")
plt.ylabel("Loss")
plt.show()

