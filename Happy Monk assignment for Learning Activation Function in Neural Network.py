#!/usr/bin/env python
# coding: utf-8

# # Feed-Forward Neural Network with two hidden layers

# In[131]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report


# In[132]:


from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.utils import np_utils


# In[133]:


# shuffle and split the data between a train and test sets 
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[134]:


# Defining hyper parameters
np.random.seed(1337)
nb_classes = 10
batch_size = 128
nb_epochs = 20 
image_size = 784 # 28 x 28


# In[135]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])


# In[136]:


print("Number of training examples :",X_train.shape[0], "and each image is of shape (%d)"%(X_train.shape[1]))
print("Number of testing examples :",X_test.shape[0], "and each image is of shape (%d)"%(X_test.shape[1])) 


# In[137]:


#Normalization
X_train = X_train/255
X_test = X_test/255


# In[138]:


#labelling

print("Class label of first image :", y_train[0])

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print("After converting the output into a vector : ", Y_train[0])


# In[ ]:





# In[140]:


#Model Building with two hidden layers

model = Sequential()

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=32, activation='relu', input_shape=(image_size,)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=nb_classes, activation='softmax'))
model.summary() 


# In[141]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[142]:


history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(X_test,Y_test)) 


# In[ ]:





# In[143]:


#evalute the model

loss, accuracy = model.evaluate(X_test,Y_test,verbose=0) 
print('Model loss: %2f, Accuracy: %2f'%((loss*100),(accuracy*100))) 


# In[144]:


#Model Prediction
y_train_predclass = model.predict_classes(X_train,batch_size=batch_size)
y_test_predclass = model.predict_classes(X_test,batch_size=batch_size)
print ("\nDeep Neural Network - Train accuracy:"), (round(accuracy_score(y_train,y_train_predclass),3))
print ("\nDeep Neural Network - Train Classification Report")
print ("classification_report(y_train,y_train_predclass)") 
print ("\nDeep Neural Network - Train Confusion Matrix\n")
print (pd.crosstab(y_train,y_train_predclass,rownames = ["Actual"],colnames = ["Predicted"]) ) 


# In[145]:


print ("\nDeep Neural Network - Testaccuracy:"),(round(accuracy_score(y_test, y_test_predclass),3))
print ("\nDeep Neural Network - Test Classification Report")
print (classification_report(y_test,y_test_predclass))
print ("\nDeep Neural Network - Test Confusion Matrix\n")
print (pd.crosstab(y_test,y_test_predclass,rownames =["Actual"],colnames = ["Predicted"]) )


# In[ ]:





# In[146]:


model.metrics_names


# In[ ]:





# In[147]:


# summarize history for loss

loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[148]:


# summarize history for accuracy

accuracy_values = history.history['accuracy']
epochs = range(1, len(accuracy_values)+1)

plt.plot(epochs, accuracy_values, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:





# In[149]:


print(history.history.keys())


# In[150]:


import time

def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label = "Validation Loss")
    ax.plot(x, ty, 'r', label = "Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[151]:


fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') 
ax.set_ylabel("categorical crossentropy loss")
x=list(range(1, nb_epochs+1))  
vy=history.history['val_loss']
ty=history.history['loss']
plt_dynamic(x, vy, ty, ax) 


# # THANK YOU, 
# 
# 
# Regards,
# 
# CHANDRA MOULI

# In[ ]:





# In[ ]:




