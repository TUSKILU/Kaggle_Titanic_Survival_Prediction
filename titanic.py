
# ------------read in data-----------------

import pandas as pd
import titanic_clean as tc

#---Preprocess-------
train_data= pd.read_csv("C:/data/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("C:/data/kaggle/input/titanic/test.csv")
train_label= train_data["Survived"]
train_data_cleaned= tc.clean_data(train_data,label=True)
test_data_cleaned = tc.clean_data(test_data, label= False)


    



"""
#----model training-------

    #------grid search find best hypervalue------------------
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV as GSCV
#Create dictionary for estimator we estimate n_estimator, max_depth of RF 
param_grid={'n_estimators':[100,150,200],'max_depth':[8,9,10,11,12]}
forest_reg=rf()
grid_search = GSCV(forest_reg,param_grid,cv=5,scoring='accuracy',return_train_score= True)
grid_search.fit(train_data_cleaned,train_label)
result= grid_search.best_params_

max_dep= result['max_depth']
n_est= result['n_estimators']
 
    #-----Random Forest----------

from sklearn.ensemble import RandomForestClassifier as rf
rnd_clf = rf(max_depth=max_dep,n_estimators=n_est)
rnd_clf.fit(train_data_cleaned,train_label)

"""
"""
#--- FeedForward Model 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import matplotlib.pyplot as plt

class titDataset(Dataset):
    def __init__(self, x, y=None):
        self.x= x
        self.y= y
        if y is not None:
            self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        X = self.x[idx]
        if self.y is not None:
            Y = self.y[idx]
            return X , Y

        else:
            return X
        
class linreg(nn.Module):
    def __init__(self):
        super(linreg, self).__init__()
        
        self.layer= nn.Sequential(
            nn.Linear(7, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        return self.layer(x)
    
train_x = train_data_cleaned[:int(len(train_data)*4/5)]
train_y = train_label[:int(len(train_data)*4/5)]

val_x  = train_data_cleaned[int(len(train_data)*4/5):]
val_y  = train_label[int(len(train_data)*4/5):]

train_set = titDataset(np.array(train_x.values,dtype=float),train_y.values)
val_set = titDataset(np.array(val_x.values,dtype=float), val_y.values)
batch =100

train_loader= DataLoader(train_set,batch_size=batch, shuffle=True) 
val_loader= DataLoader(val_set,batch_size=batch, shuffle=True)

model = linreg().cuda()
loss = nn.BCELoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.03) # optimizer 使用 Adam
num_epoch = 100   
train_acc_record=[]
val_acc_record=[]

for epoch in range(num_epoch):

    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # train mode=> enable update
    for i, data in enumerate(train_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].float().cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred.float().flatten(), data[1].float().cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
                
        train_acc += np.sum(np.round(train_pred.cpu().data.numpy()).flatten() == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()# stop updata for testing 
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].float().cuda())
            batch_loss = loss(val_pred.float().flatten(), data[1].float().cuda())

            val_acc += np.sum(np.round(val_pred.cpu().data.numpy()).flatten() == data[1].numpy())
            val_loss += batch_loss.item()
        
        train_acc_record.append(train_acc/train_set.__len__())
        val_acc_record.append(val_acc/val_set.__len__())

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

plotepochs = range(1,101)
plt.plot(plotepochs, train_acc_record, 'g', label='Training acc')
plt.plot(plotepochs, val_acc_record, 'b', label='validation acc')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()
#torch.save(model.state_dict(), 'model_weights.pth')


train_val_set = titDataset(np.array(train_data_cleaned.values,dtype=float),train_label.values)
train_val_loader = DataLoader(train_val_set,batch_size=batch, shuffle=True)


model_best = linreg().cuda()
loss = nn.BCELoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.03) # optimizer 使用 Adam
num_epoch = 40

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].float().cuda())
        batch_loss = loss(train_pred.float().flatten(), data[1].float().cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.round(train_pred.cpu().data.numpy().flatten()) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))


test_set = titDataset(np.array(test_data_cleaned.values,dtype=float))
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = torch.round(model_best(data.float().cuda()))
        print(f"test_pred size = {test_pred.size()}")
        test_label = torch.round(test_pred)
        for y in test_label:
            prediction.append(y)

with open("predict.csv", 'w') as f:
    f.write('PassengerId,Survived\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i+892,int(y[0].cpu().item())))

"""
    #---------------SVM RBF----------
        #--- find best method -----

from sklearn import svm
#create a classifier
clf = svm.SVC(kernel="rbf")
    #------grid search find best hypervalue------------------

from sklearn.model_selection import GridSearchCV as GSCV
#Create dictionary for estimator we estimate n_estimator, max_depth of RF 
param_grid={'kernel':['linear', 'poly', 'rbf']}
mysvc=svm.SVC()
grid_search = GSCV(mysvc,param_grid,cv=5,scoring='accuracy',return_train_score= True)
grid_search.fit(train_data_cleaned,train_label)
result= grid_search.best_params_

max_ker= result['kernel']
#result is linear here



#create a classifier
clf = svm.SVC(kernel="linear") 
#train the model
clf.fit(train_data_cleaned,train_label)
#predict the response


"""    
"""   
#--------out put csv file--------------------------------
import os 
def out_csv(clf):
    y_predict= clf.predict(test_data_cleaned)
    submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':y_predict})
    i=1
    while os.path.isfile('./Titanic Predictions '+str(i)+'.csv'):
        i+=1
    filename = './Titanic Predictions '+str(i)+'.csv'
    submission.to_csv(filename,index=False)
    print('Saved file: ' + filename)

out_csv(rnd_clf)

"""
