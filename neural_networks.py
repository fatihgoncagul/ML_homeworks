# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
import torch.nn 
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from load_data import load_dataset
from tag_decorator import problem




class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        self.h = h
        self.d = d
        self.k = k
        self.linear0 = 1/np.sqrt(d)
        self.linear1 = 1/np.sqrt(h)
        self.w0 = torch.FloatTensor(h,d).uniform_(-self.linear0,self.linear0)
        self.w1 = torch.FloatTensor(k,h).uniform_(-self.linear1,self.linear1)
        self.b0 = torch.FloatTensor(1,h).uniform_(-self.linear0,self.linear0)
        self.b1 = torch.FloatTensor(1,k).uniform_(-self.linear1,self.linear1)
        
        self.params = [self.w0, self.b0, self.w1, self.b1]
        for param in self.params:
            param.requires_grad = True ##recording


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        
        x =torch.matmul(x, self.w0.T) + self.b0
        x =relu(x)
        x =torch.matmul(x, self.w1.T) + self.b1
        return x



class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        self.h0 = h0
        self.h1=h1
        self.d = d
        self.k = k
        self.linear0 = 1/np.sqrt(d)
        self.linear1 = 1/np.sqrt(h0)
        self.linear2 = 1/np.sqrt(h1)

        self.w0 = torch.FloatTensor(h0, d).uniform_(-self.linear0, self.linear0)
        self.w1 = torch.FloatTensor(h1, h0).uniform_(-self.linear1, self.linear1)
        self.w2 = torch.FloatTensor(k, h1).uniform_(-self.linear2, self.linear2)
        self.b0 = torch.FloatTensor(1, h0).uniform_(-self.linear0, self.linear0)
        self.b1 = torch.FloatTensor(1, h1).uniform_(-self.linear1, self.linear1)
        self.b2 = torch.FloatTensor(1, k).uniform_(-self.linear2, self.linear2)

        self.params = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]
        for param in self.params:
            param.requires_grad = True




    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        
        x =torch.matmul(x, self.w0.T) + self.b0
        x =relu(x)
        x =torch.matmul(x, self.w1.T) + self.b1
        x =relu(x)
        x =torch.matmul(x, self.w2.T) + self.b2
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    

    losses =[]
    for i in range(32):
        loss_epc = 0
        acc = 0
        for batch in tqdm(train_loader):
            x, y = batch
            x = x.view(-1,784)
            optimizer.zero_grad()
            logist = model.forward(x)
            preds = torch.argmax(logist, 1)
            acc += torch.sum(preds == y).item()
            loss = cross_entropy(logist, y)
            loss_epc += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch:", i+1)
        print("Loss:", loss_epc / len(train_loader.dataset))
        print("Acc:", acc / len(train_loader.dataset))
        losses.append(loss_epc / len(train_loader.dataset))
        if acc / len(train_loader.dataset) > 0.99:
            break
            
    return losses


def myeval(model, test_loader):
    loss_epc = 0
    acc = 0
    for batch in tqdm(test_loader):
        x,y = batch
        x = x.view(-1, 784)

        logits = model.forward(x)
        preds = torch.argmax(logits, 1)
        acc += torch.sum(preds == y).item()
        loss = cross_entropy(logits, y)
        loss_epc += loss.item()

    los = loss_epc/len(test_loader.dataset)
    ac = acc/len(test_loader.dataset)
    return los, ac





@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    h,d,k = 64, 784, 10
    h0, h1= 32, 32

    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    
    # F1 ####################################################
    model = F1(h, d, k)
    opti = Adam(model.params, lr = 0.0001)
    losses = train(model, opti, train_loader)

    t = range(1, len(losses)+1)
    plt.plot(t, losses)
    plt.ylabel('losses')
    plt.xlabel('epoch')
    plt.title('F1 model training loss')
    plt.show()

    losses, ac = myeval(model, test_loader)
    print("test set")
    print("losses: ", losses)
    print("Accuracy: ", ac)

    F1_param = 0
    for p in model.params:
        F1_param += np.prod(p.shape)
    print("F1 model has ", F1_param, "trainable parameters.")
    print("-------------------------------------------------")


    # F2 ####################################################
    model2 = F2(h0, h1, d, k)
    opti = Adam(model2.params, lr = 0.0001)
    losses2 = train(model2, opti, train_loader)

    t2 = range(1, len(losses2)+1)
    plt.plot(t2, losses2)
    plt.ylabel('losses')
    plt.xlabel('epoch')
    plt.title('F2 model training loss')
    plt.show()

    losses, ac = myeval(model2, test_loader)
    print("test set")
    print("losses: ", losses)
    print("Accuracy: ", ac)

    F2_param = 0
    for p in model2.params: 
        F2_param += np.prod(p.shape)
    print("F2 model has ", F2_param, "trainable parameters.")
    print("-------------------------------------------------")

    


if __name__ == "__main__":
    main()
    
    #F1 model has  50890 trainable parameters.
    #F2 model has  26506 trainable parameters.