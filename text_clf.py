# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# max_features is an important parameter. You should adjust it.
vectorizer = TfidfVectorizer(max_features=420)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

print('*****')
Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)


#clf = sklearn.linear_model.LogisticRegressionCV()
#clf.fit(X_train, Y_train)



#net

class network(object):
    def __init__(self,ftrs_num,cl_num):
        '''
        input X:example_num*ftrs_num; there are cl_num different class
        '''
        self.ftrs_num=ftrs_num
        self.cl_num=cl_num
        
        W1=np.random.randn(ftrs_num,20)
        b1=np.random.randn(1,20)
        W2=np.random.randn(20,cl_num)
        b2=np.random.randn(1,cl_num)
        
        self.model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    
    def forward(self,X,y):
        self.X=X
        self.y=y
        self.example_num=X.shape[0]
        self.y_ext=np.zeros((self.example_num,self.cl_num))
        for i,j in enumerate(y,):
            self.y_ext[i,j]=1 

        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        self.exp_scores=np.exp(self.z2)
        self.probs=self.exp_scores/np.sum(self.exp_scores,axis=1,keepdims=True)
        loss=-(self.y_ext*np.log(self.probs)).sum()/self.example_num
        predict=np.argmax(self.probs,axis=1)
        return loss, predict
        
    
    def backward(self, X, y, reg_lambda=0.0001,epsilon=0.0016):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        delta3=self.probs
        delta3[range(self.example_num),self.y]-=1
        dW2=(self.a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))

        dW1=(X.T).dot(delta2)
        db1=np.sum(delta2,axis=0)
        
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        #W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        #dX = [dW1, db1, dW2, db2]
        
        W1+= -epsilon*dW1
        b1+= -epsilon*db1
        W2+= -epsilon*dW2
        b2+= -epsilon*db2
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']=W1,b1,W2,b2
        

if __name__=='__main__':
       net=network(420,4)
       iter = 0
       acc = 0 
       
       while iter <= 5000:
            loss, Y_predict= net.forward(X_train, Y_train)
            if iter % 100 == 0:
                print("iter: %d" % iter)
                print("loss: %f" % loss)
            
            #Y_predict = clf.predict(X_test)
            #acc = (Y_predict == Y_train).sum() / len(Y_train)
            #print('1111111111')
            net.backward(X_train, Y_train)
            #print('3333333333')
            iter += 1
            
       #print(Y_test)
       #print(Y_predict)
 
       loss , Y_predict = net.forward(X_test, Y_test)
       ncorrect = 0
       for dy in  (Y_test - Y_predict):
	          if 0 == dy:
		            ncorrect += 1

       print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test)) ) )


