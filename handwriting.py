import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

read_data_pd_training = pd.read_csv('./letter_recognition_training_data_set.csv')
read_data_pd_testing = pd.read_csv('./letter_recognition_testing_data_set.csv')
ohv_data = pd.get_dummies(read_data_pd_training,sparse=True)
# ovh_test_data = pd.get_dummies(read_data_pd_testing,sparse=True)
#training data set
train_data = ohv_data.iloc[:16000,:16]
train_label = ohv_data.iloc[:16000,-26:]
#training length
train_length = len(train_data)

#simulate testing data set
test_data = ohv_data.iloc[-4000:,:16]
test_label = ohv_data.iloc[-4000:,-26:]


# Parameters
learning_rate = 0.008
training_epochs = 15
batch_size = 1000
display_step = 1

# Network Parameters
# n_hidden_0 = 256 # 0 layer
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 16 # data input
n_classes = 26 #total classes (A-Z letters)

# reset the graph
tf.reset_default_graph()

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # #Hidden layer with Sigmod avtivation
    # layer_0 = tf.add (tf.matmul (x, weights['h0']), biases['b0'])
    # layer_0 = tf.nn.sigmoid(layer_0)
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    # 'h0': tf.Variable(tf.random_normal([n_input, n_hidden_0])),
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    # 'b0': tf.Variable(tf.random_normal([n_hidden_0])),
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_length/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: train_data.iloc[i*batch_size:(i+1)*batch_size-1],
                                                         y: train_label.iloc[i*batch_size:(i+1)*batch_size-1]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    test_value = tf.cast(correct_prediction, "float")
    # print(correct_prediction)
    # print(sess.run(correct_prediction,feed_dict={x}))
    # print(test_value)
    # print(test_value.shape)

    # Calculate accuracy
    accuracy = tf.reduce_mean(test_value)
    #for each capital letters accuracy value
   


    print("Accuracy:",accuracy.eval({x: test_data, y: test_label}))
    print ('-------------- prediction ----------------')
    prediction_value = sess.run(pred, feed_dict={x:read_data_pd_testing})
    a = tf.argmax(prediction_value,1)
    # print (a.eval())
    temp = a.eval()
    result = np.bincount(temp)
    # print (result.sum())
    print("the numeber of A-Z letters in the testing set is")
    print(result)
    # ohv_prediction_value = pd.get_dummies(prediction_value,sparse=True)
    # print (ohv_prediction_value)
    # for i in range(len(prediction_value)):
    #      print(prediction_value[i])
    # X= sum0
    # pca = PCA (n_components=1)
    # pca_transf = pca.fit_transform (X)
    # print (pca_transf)
    # scale = StandardScaler()
    # scale_transf = scale.fit_transform(prediction_value)
    # print (scale_transf)


    # draw the figure to show the numbers for each letter in the testing dataset
    def draw_bar(labels, quants):
        width = 0.4
        ind = np.linspace (0, 26, 26)
        # make a square figure
        fig = plt.figure (1)
        ax = fig.add_subplot (111)
        # Bar Plot
        ax.bar (ind - width / 2, quants, width, color='green')
        # Set the ticks on x-axis
        ax.set_xticks (ind)
        ax.set_xticklabels (labels)
        # labels
        ax.set_xlabel ("A-Z")
        ax.set_ylabel ("numbers in the test set")
        # title
        ax.set_title ('ALL LETTERS NUMBERS', bbox={'facecolor': '0.8', 'pad': 5})
        plt.grid (True)
        plt.show ()
        plt.savefig ("/Users/zhangyangzuo/Downloads/capture/lettersbarchat.png")
        plt.close ()


    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

    draw_bar (labels, result)
