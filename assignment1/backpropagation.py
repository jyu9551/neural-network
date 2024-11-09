import numpy as np
import requests, gzip, os, hashlib
import struct as st
import matplotlib.pyplot as plt

"""
#fetch data
path='./data'
def fetch(url):
  fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
     with open(fp, "rb") as f:
        data = f.read()
  else:
     with open(fp, "wb") as f:
        data = requests.get(url).content
        f.write(data)
  return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

train_X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
train_y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
"""


## 1. read data
def read_input(file):
    """

    Args:
        file (idx): binary input file.

    Returns:
        numpy: arrays for our dataset.

    """

    with open(file, 'rb') as file:
        z, d_type, d = st.unpack('>HBB', file.read(4))
        shape = tuple(st.unpack('>I', file.read(4))[0] for d in range(d))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


train_size = 60000      
test_size = 10000      
input_size = 28*28
hidden_neurons = 150
output_neurons = 10

X = read_input("mnist/train-images.idx3-ubyte")         # (60000,28,28)
Y = read_input("mnist/train-labels.idx1-ubyte")         # (60000,)
X_test = read_input("mnist/t10k-images.idx3-ubyte")           # (10000,28,28)
Y_test = read_input("mnist/t10k-labels.idx1-ubyte")           # (10000, )

print(X.shape, Y.shape, X_test.shape, Y_test.shape)
print("Y_test : ", Y)
## 2. Train / Validation split

# (Train)
rand=np.arange(60000)
np.random.shuffle(rand)
train_no=rand[:50000]

# (Validation)
val_no=np.setdiff1d(rand,train_no)

X_train, X_val = X[train_no,:,:], X[val_no,:,:]
Y_train, Y_val = Y[train_no], Y[val_no]
print(X_train.shape, X_val.shape)

## 3. define function (sig, softmax, init)

# (Sigmoid and its derivative)
def sigmoid(z):
  return 1/(np.exp(-z)+1)
def d_sigmoid(z):
  return (np.exp(-z))/((np.exp(-z)+1)**2)


# (Softmax)
def softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)
def d_softmax(a):
  exp_element=np.exp(a-a.max())
  return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
   
# (Initializing weights)
def init(x,y):
  layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
  return layer.astype(np.float32)
np.random.seed(42)
l1=init(28*28,128)  # layer 1 (784,128)
l2=init(128,10)     # layer 2 (128,10)

print(l1.shape, l2.shape)


## 4. forward and backward pass
def forward_backward_pass(x,y): # (128,784) (128, )
  targets = np.zeros((len(y),10), np.float32) # (128, 10)
  #print("targets:", targets.shape)
  targets[range(targets.shape[0]),y] = 1 # = targets[range(128), (128,)] = 1
  #print(targets, y)

  x_l1p=x.dot(l1)             # (128 128) = (128 784) (784 128)
  x_sigmoid=sigmoid(x_l1p)    # (128 128) = sig(128 128)
  x_l2p=x_sigmoid.dot(l2)     # (128 10) = (128 128) (128 10)
  out=softmax(x_l2p)          # (128 10) = softmax(128 10)
  #print("out:", out.shape)

  error=2*(out-targets)/out.shape[0]*d_softmax(x_l2p)
  update_l2=x_sigmoid.T@error
  print("error, updata_l2 shape :", error.shape, update_l2.shape) # (128 10) (128 10)
    
    
  error=((l2).dot(error.T)).T*d_sigmoid(x_l1p)
  update_l1=x.T@error
  print("error, updata_l1 shape :", error.shape, update_l1.shape) # (128 128) (784 128)

  return out,update_l1,update_l2 


## 5. Training
epochs = 10000  # num of epochs
lr = 0.001      # Learning Rate
batch = 128     # batch size

losses,accuries,val_accuracies=[],[],[]

for i in range(epochs):
    #randomize and create batches
    sample=np.random.randint(0,X_train.shape[0],size=(batch))
    x=X_train[sample].reshape((-1,28*28))
    y=Y_train[sample]
    #print("x y shape:", x.shape, y.shape) # (64,256) (64,)


    out,update_l1,update_l2=forward_backward_pass(x,y)   
    category=np.argmax(out,axis=1)
    
    accuracy=(category==y).mean()
    accuries.append(accuracy.item())
    
    loss=((category-y)**2).mean()
    losses.append(loss.item())
    
    #SGD 
    l1=l1-lr*update_l1
    l2=l2-lr*update_l2


    #testing our model using the validation set every 20 epochs
    if(i%20==0):    
      X_val=X_val.reshape((-1,28*28))
      val_out=np.argmax(softmax(sigmoid(X_val.dot(l1)).dot(l2)),axis=1)
      val_acc=(val_out==Y_val).mean()
      val_accuracies.append(val_acc.item())
    if(i%1000==0): print(f'For {i}th epoch ) train loss: {loss:.3f} train acc: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')


X_test = X_test.reshape((-1,28*28))
test_out = np.argmax(softmax(sigmoid(X_test.dot(l1)).dot(l2)),axis=1) # ValueError: shapes (10000,28,28) and (784,128) not aligned: 28 (dim 2) != 784 (dim 0)
test_acc=(test_out==Y_test).mean().item()
test_accuries = []; test_accuries.append(test_acc)
print(f'Test accuracy = {test_acc:.4f}')
np.savez('weights',l1,l2)


plt.figure(figsize=(10, 5))
plt.plot(range(len(accuries)),
         accuries, 'b', label='Train Data', linewidth=0.3)
#plt.plot(range(len(test_accuries)), test_accuries, 'r', label='Test Data')
#plt.axhline(y=500, c='black', linestyle=':')
plt.title('Accuracy vs Epoch')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
