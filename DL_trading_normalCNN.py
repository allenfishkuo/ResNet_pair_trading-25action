# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:16:03 2020

@author: allen
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import dataloader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import test 
import test_profit
# Hyper Parameters
EPOCH = 100       # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1024
INPUT_SIZE = 150         # rnn input size / image width
LR = 0.01               # learning rate

#train_data, train_label, test_data, test_label = dataloader.read_data()
def DataTransfer(train_data, train_label, test_data, test_label):
    
    train_data = torch.FloatTensor(train_data).cuda()
    train_label=torch.LongTensor(train_label).cuda() #data型態轉換@
    test_data=torch.FloatTensor(test_data).cuda()
    test_label=torch.LongTensor(test_label).cuda()
    torch_dataset_train = Data.TensorDataset(train_data, train_label)
    loader_train = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = BATCH_SIZE,      # mini batch size
            shuffle = True,               
            )

    torch_dataset_test = Data.TensorDataset(test_data, test_label)
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle = False,              
            )
    return loader_train, loader_test

                     # the target label is not one-hotted

#loader_train,loader_test= DataTransfer(train_data, train_label, test_data, test_label)
class CNN_classsification1(nn.Module):
    def __init__(self):
        super(CNN_classsification1,self).__init__()
        #activations=nn.ModuleDict([['ELU',nn.ELU(alpha=1.0)],['ReLU',nn.ReLU()],['LeakyReLU',nn.LeakyReLU()]])
        self.Conv1=nn.Sequential(
                nn.Conv1d(in_channels=3,out_channels=25,kernel_size=5,stride=1,bias=False),
                #nn.Conv1d(25,25, kernel_size=10, stride=1,bias=False),
                nn.BatchNorm1d(25,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                #activations[activation_function],
                nn.LeakyReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(p=0.5)
                )
        
        self.Conv2=nn.Sequential(
                 nn.Conv1d(25,50,kernel_size=5,stride=1,bias=False),
                 #nn.Conv1d(50,100,kernel_size = 5, stride = 1,bias =False),
                 nn.BatchNorm1d(50,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                 #activations[activation_function],
                 nn.LeakyReLU(),
                 
                 nn.MaxPool1d(2),
                 nn.Dropout(p=0.3)
                 
                 )   
        
        self.Conv3=nn.Sequential(
                 nn.Conv1d(50,100,kernel_size=5,stride=1,bias=False),
                 nn.BatchNorm1d(100,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                 #activations[activation_function],
                 nn.LeakyReLU(),
                 nn.MaxPool1d(kernel_size=2),
                 nn.Dropout(p=0.3)
                 )
        self.Conv4=nn.Sequential(
                nn.Conv1d(100,200,kernel_size=5,stride=1,bias=False),
                nn.BatchNorm1d(200,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                #activations[activation_function],
                nn.LeakyReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(p=0.2)
        
                )
        
        self.classify=nn.Sequential(
                 nn.Linear(in_features=5600,out_features=25,bias=True),
                 #nn.LogSoftmax()
                 )
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.classify(x)
        #print(output)
        return output



# training and testing
def model_train(loader_train,loader_test):
    
    
    CNN1_class = CNN_classsification1().cuda()
    optimizer = torch.optim.Adam(CNN1_class.parameters(), lr=LR)   # optimize all cnn parameters
    loss_xe = nn.CrossEntropyLoss().cuda()          
   
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 60, 75, 90], gamma=0.1)
    
    train_loss =[]
    train_acc=[]
    big_profit  = -10
    save_loss = 1000
    test_acc = 0
    for epoch in range(EPOCH):
            total_train = 0
            correct_train = 0
            action_list=[]
            action_choose=[]
            CNN1_class.train()
            #scheduler.step()
            
            for step, (batch_x, batch_y) in enumerate(loader_train):   # 分配 batch data, normalize x when iterate train_loader
                output = CNN1_class(batch_x)               # cnn output
                loss = loss_xe(output, batch_y)   # mseloss
                
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                _, predicted = torch.max(output, 1)
                #print("預測分類 :",predicted)
                total_train += batch_y.nelement()
                correct_train += predicted.eq(batch_y).sum().item()
            train_accuracy = 100 * correct_train / total_train 
            train_loss.append(loss.item())                                          
            print('Epoch {}, train Loss: {:.5f}'.format(epoch+1, loss.item()), "Training Accuracy: %.2f %%" % (train_accuracy))
            
            train_acc.append(train_accuracy)

            for step, (batch_x, batch_y) in enumerate(loader_test):
                output = CNN1_class(batch_x).cuda()
                loss = loss_xe(output, batch_y).cuda()
                _, predicted = torch.max(output, 1)
                action_choose = predicted.cpu().numpy()
                action_choose = action_choose.tolist()
                action_list.append(action_choose)
            action_list =sum(action_list, [])
            print("幾個 action :",len(action_list))
            profit = test_profit.reward(action_list)
            #test_accuracy = 100 * correct_test / total_test          #avg_accuracy = train_accuracy / len(train_loader)
            print('Epoch {}, test Loss: {:.5f}'.format(epoch+1, loss.item()), "Testing winrate: %.2f %%" % (profit))
            if train_accuracy >= 5:
                print("in")
                torch.save(CNN1_class,"normalCNN_2017-2018.pkl")
    draw(train_loss,train_acc)
def draw(train_loss, train_acc):
    plt.title('CNN_Net Classificaiton for pair_trading')
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.plot(train_loss)
    plt.show()        
    plt.close()

    plt.title("CNN_Net Classificaiton for pair_trading")
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuaracy(%)')
    plt.plot(train_acc)
    plt.show()        
    plt.close()
#print(action_choose)



if __name__=='__main__':
    choose = 1
    if choose == 0 :
        train_data, train_label, test_data, test_label = dataloader.read_data()
        loader_train,loader_test = DataTransfer(train_data, train_label, test_data, test_label)
        model_train(loader_train,loader_test)
    else :
    #model_train()
        
        test.test_reward()