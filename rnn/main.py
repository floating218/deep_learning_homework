
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn
import matplotlib.pylab as plt


# In[2]:


# Feel free to add any functions, import statements, and variables.
def result_rnn(Xtest):
    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: Xtest})
    return test_predict

def predict(file): #file=test.csv
    testX=pd.read_csv(file)
    testX=testX.drop(['Date'],axis=1)
    testX=np.array(testX)
    testX=testX.reshape(testX.shape[0],6,1)
    result=result_rnn(testX)
    result=[i for i in result.flatten()]
    return result 


def write_result(predictions):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Value\n')
        for l in predictions:
            f.write('{}\n'.format(l))


def main():
    # You don't need to modify this function.
    predictions = predict('test.csv')
    print(predictions)
    write_result(predictions)



# In[3]:


# train Parameters
seq_length = 6 #7->6
data_dim = 1 #5->1
hidden_dim = 6 #10->6 내가 정함
output_dim = 1
learning_rate = 0.01
#iterations = 500


# In[4]:


# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim]) 
#None, 6개인풋(시퀀스), 1차원

Y = tf.placeholder(tf.float32, [None, 1])
#None, 1개아웃풋

cells=[]
for _ in range(1):
    cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) #output이 timestep숫자만큼(sequencelenth) 나옴
    cells.append(cell)
cell=tf.contrib.rnn.MultiRNNCell(cells)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output


# cost/loss
loss = tf.reduce_mean(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
sess=tf.Session()

init = tf.global_variables_initializer()
sess.run(init)


# In[5]:


df_train=pd.read_csv('train.csv')


def crossval(df_train):
    Xtr=df_train.drop(['Date','Label'],axis=1)
    Xtr=np.array(Xtr) #3279*6
    ytr=df_train['Label']
    Xto=Xtr
    yto=ytr
    ytr=np.array(ytr) #3289
    #Xtr, Xva, ytr, yva = Xtr[:-300],Xtr[-300:],ytr[:-300],ytr[-300:]#random_state
    Xtr, Xva, ytr, yva = Xtr[200:],Xtr[:200],ytr[200:],ytr[:200]#random_state
    Xtr=Xtr.reshape(Xtr.shape[0],6,1)
    ytr=ytr.reshape(Xtr.shape[0],1)
    Xto=Xto.reshape(Xto.shape[0],6,1)
    yto=yto.reshape(Xto.shape[0],1)
    Xva=Xva.reshape(Xva.shape[0],6,1)
    yva=yva.reshape(Xva.shape[0],1)
    return Xto,yto,Xtr,ytr,Xva,yva

Xto,yto,Xtr,ytr,Xva,yva=crossval(df_train)


# In[6]:


def train_rnn(iternumber,Xtr,ytr):
    # Training step
    for i in range(iternumber):
        Xtr=Xtr.reshape(Xtr.shape[0],6,1)
        ytr=ytr.reshape(Xtr.shape[0],1)
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: Xtr, Y: ytr})
        print("[step: {}] loss: {}".format(i, step_loss))

train_rnn(3000,Xto,yto)


# In[7]:


def test_rnn(Xva,yva):
    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: Xva})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: yva, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(yva,'b-')
    plt.plot(test_predict,'r-')
    plt.xlabel("Time Period")
    plt.ylabel("Temperature")
    plt.show()
    
#test_rnn(Xva,yva)


# In[ ]:





# In[8]:



if __name__ == '__main__':
    # You don't need to modify this part.
    main()

