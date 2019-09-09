import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib 
 
# Display plots inline and change default figure size 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
 

np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20) 
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 
 

# Train the logistic rgeression classifier 
clf = sklearn.linear_model.LogisticRegressionCV() 
clf.fit(X, y) 
 
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
        #print(W1.shape)
        #print(X.shape)
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        self.exp_scores=np.exp(self.z2)
        self.probs=self.exp_scores/np.sum(self.exp_scores,axis=1,keepdims=True)
        loss=-(self.y_ext*np.log(self.probs)).sum()/self.example_num
        #predict=np.argmax(self.probs,axis=1)
        return loss
        
    
    def backward(self, X, y, epsilon=0.001):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        delta3=self.probs
        delta3[range(self.example_num),self.y]-=1
        dW2=(self.a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))
        dW1=np.dot(X.T,delta2)
        db1=np.sum(delta2,axis=0)
        
        W1+=-epsilon*dX[0]
        b1+=-epsilon*dX[1]
        W2+=-epsilon*dX[2]
        b2+=-epsilon*dX[3]
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']=W1,b1,W2,b2

# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below.

def predict(model,x):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=x.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)
    
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 

# Plot the decision boundary 
if __name__=='__main__':
       net=network(2,2)
       i = 0 
       acc = 0 
       
       while i <= 1000:
            loss = net.forward(X, y)
            
            #Y_predict = clf.predict(X_test)
            #acc = (Y_predict == Y_train).sum() / len(Y_train)
            net.backward(X, y)
            i += 1
       model = net.model
       plot_decision_boundary(lambda x: predict(model,x))
       plt.title("Logistic Regression") 
       plt.show()
