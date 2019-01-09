import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import csv


##init training data
print("!  Loading training data")
FEATURE_CLASSES = ['Pclass','Age', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Survived']

filename = "train.csv"
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=",", quoting=csv.QUOTE_NONE)

x = list(reader)
data = np.array(x)
data = np.delete(data, 0)
Ys = []
print(data[0])
print(data[1])

for row in data:
    
    row.pop(0)
    row.pop(2)
    row.pop(2)
    row.pop(6)
    row.pop(7)

    

    Ys.append(row[0])
    row.pop(0)

    male = int(row[1] == 'male')
    row[1] = int(male)
    row.append(int(not male))
    
    for i in range(len(row)):
        try:
            row[i] = float(row[i])
        except:
            row[i] = 0

    embarked = row[7]

    if embarked == "C":
        row[7] = 1
    else:
        row[7] = 0
    if embarked == "S":
        row.append(1)
    else:
        row.append(0)
    if embarked == "Q":
        row.append(1)
    else:
        row.append(0)
    row = np.asarray(row)

ys = np.zeros((len(data), 2))
xs = np.zeros((len(data), 10))


for row in range(len(data)):
    for col in range(len(data[row])):
        xs[row][col] = data[row][col]
    if int(Ys[row]) == 0:        
        ys[row][0] = 1
        ys[row][1] = 0
    else:
        ys[row][0] = 0
        ys[row][1] = 1

print(xs)
print(ys)

print("!  Loaded training data")

##neural network parameters
learning_rate = 0.001
epochs = 20000
batch_size = 1

x = tf.placeholder(tf.float32, [None, 10])
y = tf.placeholder(tf.float32, [None, 2])




##tensorflow neural network
print("!  Initializing neural network")

W1 = tf.Variable(tf.random_normal([10, 20], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([20]), name='b1')

W2 = tf.Variable(tf.random_normal([20, 20], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([20]), name='b2')

W3 = tf.Variable(tf.random_normal([20, 2], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([2]), name='b3')

##tensorflow neural networks calculations
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

hidden_out = tf.add(tf.matmul(hidden_out, W2), b2)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W3), b3))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

##add an optimiser for gradient descend
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

print("!  Initialized neural network")

##starting training
print("!  Starting training")
init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##training
with tf.Session() as sess:
    #initialize tensorflow
    sess.run(init_op)
    total_batch = int(len(xs)/batch_size)
    batches_x = np.array_split(xs, total_batch)
    batches_y = np.array_split(ys, total_batch)
    
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = batches_x[i], batches_y[i]
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y:batch_y})
            avg_cost += c / (total_batch)
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print("!  Testing On Actual Data")
    filename = "test.csv"
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=",", quoting=csv.QUOTE_NONE)

    x_test = list(reader)
    data_test = np.array(x_test)
    data_test = np.delete(data_test, 0)
    print(data_test)
    for row in data_test:
        
        row.pop(0)
        row.pop(1)
        row.pop(1)
        row.pop(5)
        row.pop(6)



        male = int(row[1] == 'male')
        row[1] = int(male)
        row.append(int(not male))


        embarked = row[6]

        if embarked == "C":
            row[6] = 1
        else:
            row[6] = 0
        if embarked == "S":
            row.append(1)
        else:
            row.append(0)
        if embarked == "Q":
            row.append(1)
        else:
            row.append(0)
        
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except:
                row[i] = 0

        row = np.asarray(row)
    xs_test = np.zeros((len(data), 10))
    for row in range(len(data_test)):
        for col in range(len(data_test[row])):
            xs_test[row][col] = data_test[row][col]
    
    predicted_y = sess.run(y_clipped, feed_dict={x: xs_test})
    predicted_y = np.argmax(predicted_y, axis=1)
    print(predicted_y.tolist())

    submitstring = "PassengerId,Survived\n"
    pId = 892
    for datapoint in predicted_y.tolist():
        submitstring += str(pId) + "," + str(datapoint) + '\n'
        pId += 1

    print(submitstring)
    


        
    


