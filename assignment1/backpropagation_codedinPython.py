import numpy as np


import matplotlib.pyplot as plt
from PIL import Image
import os


## 1. read data

def read_dataset(file, numofdata):
    X = []  
    for i in range(1,numofdata+1):
        image_path = os.path.join(file, f"{i}.jpg")
        img = Image.open(image_path)
        img_array = np.array(img)
        X.append(img_array)
        print(i, np.array(img_array).shape)
    
    X = np.array(X)
    return X

def read_label(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        csv_string = file.read()
    csv_string = csv_string.lstrip('\ufeff').replace('"', '')
    values = [int(x) for x in csv_string.split(',')]
    Y = np.array(values)
    return Y

def read_test_dataset(file):
    X = []
    for i in range(1,21):
        image_path = os.path.join(file, f"test{i}.jpg")
        img = Image.open(image_path)
        img_array = np.array(img)
        X.append(img_array)
        print("test case", i, ")", np.array(img_array).shape)  
    
    X = np.array(X)
    return X

numTrainingSets = 420
train_size = 380 
validation_size = 40 
test_size = 20    

input_size = 16*16
hidden_neurons = 128
output_neurons = 7


X = read_dataset('train_dataset/', 420)     # train data
Y = read_label("train_label.csv")           # train label
X_test = read_dataset('test_dataset/', 20)  # test data
Y_test = read_label("test_label.csv")       # train label

print("X, Y shape :", X.shape, Y.shape)
print("X_test, Y_test shape :", X_test.shape, Y_test.shape)


## 2. Shuffle & Train, Validation Split

# (Shuffle)
rand = np.arange(numTrainingSets) # [0 ... 419]
np.random.shuffle(rand)           # shuffle

# (split)
train_no = rand[:380]   # (Train)
val_no = rand[380:]     # (Validation)

X_train, X_val = X[train_no,:,:], X[val_no,:,:]
Y_train, Y_val = Y[train_no], Y[val_no]

print("X_train, X_val shape :", X_train.shape, X_val.shape)
print("Y_train, Y_val shape :", Y_train.shape, Y_val.shape)


## 3. define function (sig, softmax, init)

# (Sigmoid and its derivative)
def sigmoid(x):
  return 1 / (np.exp(-x+1))
def d_sigmoid(x):
  return (np.exp(-x)) / ((np.exp(-x)+1)**2)


# (Softmax)
def softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)
def d_softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
   
# (Initializing weights)
def init(x,y):
  weight = np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
  return weight.astype(np.float32)
np.random.seed(42)  


w1 = init(16*16, 128)                 # layer 1 (256,128)
w2 = init(128, output_neurons)        # layer 2 (128,7)
print(w2)
#print("l1, l2, l3 shape : ", l1.shape, l2.shape, l3.shape)

## 4. forward and backward pass
def forward_backward_pass(x,y):
  # (2) targets
  targets = np.zeros((len(y),7), np.float32) 
  targets[range(targets.shape[0]),y] = 1
  
  # (3) forward pass
  z1 = x.dot(w1)    
  a1 = sigmoid(z1)    
  z2 = a1.dot(w2)     
  out = softmax(z2)

  # (4) backpropagation (MSE)
  error = 2*(out-targets)/out.shape[0]*d_softmax(z2)
  d_w2 = a1.T@error
  error = ((w2).dot(error.T)).T*d_sigmoid(z1)
  d_w1 = x.T@error

  return out, d_w1, d_w2 

## 5. Training
epochs = 18000  # num of epochs
lr = 0.001      # learning rate
batch = 128     # batch size

losses,accuries,val_accuracies=[],[],[]

for i in range(epochs):
    # (1)
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x = X_train[sample].reshape((-1, 16*16))
    y = Y_train[sample]

    out, d_w1, d_w2 = forward_backward_pass(x,y)   

    # (5) calculate loss, accuracy
    category = np.argmax(out, axis=1)
    
    accuracy = (category==y).mean()
    accuries.append(accuracy.item())
    loss = ((category-y)**2).mean()
    losses.append(loss.item())
    
    # (6) SGD (gradient descent)
    w1 = w1-lr*d_w1
    w2 = w2-lr*d_w2

    # (7) testing our model using the validation set every 20 epochs
    if(i%20==0):    
      X_val = X_val.reshape((-1, 16*16))
      val_out = np.argmax(softmax(sigmoid(X_val.dot(w1)).dot(w2)), axis=1)
      val_acc = (val_out==Y_val).mean()
      val_accuracies.append(val_acc.item())
    if(i%1000==0): print(f'For {i}th epoch ) train loss: {loss:.3f} train acc: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')


## 6. Testing
X_test = X_test.reshape((-1, 16*16))
test_out = np.argmax(softmax(sigmoid(X_test.dot(w1)).dot(w2)), axis=1)

letter_mapping = {
    0: 't',
    1: 'u',
    2: 'v',
    3: 'w',
    4: 'x',
    5: 'y',
    6: 'z',
}
test_result = [letter_mapping[num] for num in test_out]
test_target = [letter_mapping[num] for num in Y_test]

for i in range(20):
   print("expected:", test_target[i], "result:", test_result[i])


test_acc=(test_out==Y_test).mean().item()
test_accuries = []; test_accuries.append(test_acc)

print(f'Test accuracy = {test_acc:.4f}')
np.savez('weights', w1, w2)

plt.figure(figsize=(15, 5))
plt.plot(range(len(accuries)),
         accuries, 'b', label='Train Data', linewidth=0.3)
plt.title('Accuracy vs Epoch'); plt.xlabel('Number of Epochs'); plt.ylabel('Accuracy')
plt.legend(); plt.show()