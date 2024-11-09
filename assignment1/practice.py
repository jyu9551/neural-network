import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os


## 1. read data

def read_train_dataset(file):
    X = []  #
    for i in range(1,421):
        image_path = os.path.join(file, f"{i}.jpg")
        img = Image.open(image_path)
        img_array = np.array(img)
        X.append(img_array)
        print(i, np.array(img_array).shape)
    
    X = np.array(X)
    return X
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
def read_label(file_path):
# 파일 열기 및 데이터 읽기
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        csv_string = file.read()

    csv_string = csv_string.lstrip('\ufeff').replace('"', '')
    # 쉼표로 문자열을 분할하고 각 값을 정수로 변환하여 리스트에 저장
    values = [int(x) for x in csv_string.split(',')]

    # 리스트를 넘파이 어레이로 변환
    Y = np.array(values)
    return Y

numTrainingSets = 420
train_size = 380 
validation_size = 40 
test_size = 20    

input_size = 16*16
hidden_neurons = 128
output_neurons = 7


X = read_train_dataset('train_dataset/')
Y = read_label("train_label.csv")
X_test = read_test_dataset('test_dataset/')
Y_test = read_label("test_label.csv")

print("X, Y shape :", X.shape, Y.shape)
print("X_test, Y_test shape :", X_test.shape, Y_test.shape)


## 2. Shuffle & Train, Validation Split

# (Train)
rand = np.arange(numTrainingSets)
np.random.shuffle(rand)
train_no = rand[:380]

# (Validation)
val_no=np.setdiff1d(rand,train_no)

X_train, X_val = X[train_no,:,:], X[val_no,:,:]
Y_train, Y_val = Y[train_no], Y[val_no]
print("X_train, X_val shape :", X_train.shape, X_val.shape)
print("Y_train, Y_val shape :", Y_train.shape, Y_val.shape)


## 3. define function (sig, softmax, init)

# (Sigmoid and its derivative)
def sigmoid(z):
  return 1 / (np.exp(-z)+1)
def d_sigmoid(z):
  return (np.exp(-z)) / ((np.exp(-z)+1)**2)


# (Softmax)
def softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)
def d_softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
   
# (Initializing weights)
def init(x,y):
  layer = np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
  return layer.astype(np.float32)
np.random.seed(42)

w1 = init(16*16, 128)                 # layer 1 (256,128)
w2 = init(128, output_neurons)        # layer 2 (128,7)
print(w2)
#print("l1, l2, l3 shape : ", l1.shape, l2.shape, l3.shape)


## 4. forward and backward pass
def forward_backward_pass(x,y): # (128,256) (128, )
 
  return out, d_w1, d_w2 

## 5. Training
epochs = 18000  # num of epochs
lr = 0.001      # learning rate
batch = 128     # batch size

losses,accuries,val_accuracies=[],[],[]

for i in range(epochs):
    
    #randomize and create batches
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x=X_train[sample].reshape((-1, 16*16))
    y=Y_train[sample]
    #print("x y shape:", x.shape, y.shape) # (128,256) (128,)


    #back, forward
    targets = np.zeros((len(y),7), np.float32)  # (128,7)
    targets[range(targets.shape[0]),y] = 1
  
    # forward pass
    z1 = x.dot(w1)    # (128,128) = (128,256) (256,128)
    a1 = sigmoid(z1)    
    z2 = a1.dot(w2)     
    out  = softmax(z2)       

    # backpropagation
    error = 2*(out-targets)/out.shape[0]*d_softmax(z2)
    d_w2 = a1.T@error
    error = ((w2).dot(error.T)).T*d_sigmoid(z1)
    d_w1 = x.T@error
   
    category = np.argmax(out, axis=1)
    
    accuracy = (category==y).mean()
    accuries.append(accuracy.item())
    
    loss = ((category-y)**2).mean()
    losses.append(loss.item())
    
    #SGD 
    w1 = w1-lr*d_w1
    w2 = w2-lr*d_w2
    

    #testing our model using the validation set every 20 epochs
    if(i%20==0):    
      X_val = X_val.reshape((-1, 16*16))
      val_out = np.argmax(softmax(sigmoid(X_val.dot(w1)).dot(w2)), axis=1)
      val_acc = (val_out==Y_val).mean()
      val_accuracies.append(val_acc.item())
    if(i%1000==0): print(f'For {i}th epoch ) train loss: {loss:.3f} train acc: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')


X_test = X_test.reshape((-1, 16*16))
test_out = np.argmax(softmax(sigmoid(X_test.dot(w1)).dot(w2)), axis=1)

alphabet_mapping = {
    0: 't',
    1: 'u',
    2: 'v',
    3: 'w',
    4: 'x',
    5: 'y',
    6: 'z',
}
test_result = [alphabet_mapping[num] for num in test_out]
test_target = [alphabet_mapping[num] for num in Y_test]

for i in range(20):
   print("expected:", test_target[i], "result:", test_result[i])


test_acc=(test_out==Y_test).mean().item()
test_accuries = []; test_accuries.append(test_acc)

print(f'Test accuracy = {test_acc:.4f}')
np.savez('weights', w1, w2)

plt.figure(figsize=(15, 5))
plt.plot(range(len(accuries)),
         accuries, 'b', label='Train Data', linewidth=0.3)
plt.title('Accuracy vs Epoch')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()