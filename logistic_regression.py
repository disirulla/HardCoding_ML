from tinydrop import tinydrop_main_class as tmc
import numpy as np


class logistic_regression(tmc):
    '''
    Logistic Regression\n
    Classification Model\n

    Input:
    xtrain: Feature Matrix\n
    Input should be of form: Features along rows
    ytrain: Matrix containing label values\n
    Similarly xtest and ytest\n
    learning_rate\n
    #. of iterations\n

    Various functions:

    train(): To train the model\n
    validate(): To validate the results\n
    cost is the performance metric\n


    Output:
    ypred = Predicted values (Can retrieve after validation)\n
    cost = performance metric\n
    parameters = Weights\n


    Author:
    Alluri L S V Siddhartha Varma\n


    '''

    def __init__(self):
        self.w = None
        self.b = None
        self.cost = None
        self.ypred = None

    
    def sigmoid(self, z):
        '''
        Sigmoid Function
        '''
        return 1/(1+np.exp(-z))

    

    def train(self, xtrain, ytrain, learning_rate, iterations):
        '''
        Train

        Input: 
        xtrain: Feature Matrix of train Set\n
        ytrain: Label Matrix of train Set\n
        Learning rate\n
        No. of iterations\n 

        Output:
        Trained model\n
       '''
        x = xtrain
        y = ytrain.reshape(-1,1)
        learning_rate = learning_rate
        iterations = iterations
        if (len(x)!= len(y)):
            raise TinyDropError('Feature and Label matrix have different lenghts')
        
        m = x.shape[0]
        n = x.shape[1]
        self.w = np.zeros((n,1))
        self.b = 0
        for i in range(iterations):
            self.w = self.w - ((learning_rate/m) * np.dot(x.T, (self.sigmoid((x @ self.w)+ self.b)) - y))
            self.b = self.b - ((learning_rate/m) * np.sum((self.sigmoid((x @ self.w)+ self.b)) - y))

    def validate(self, xtest = None, ytest = None, set = 'test'):
        '''
        Validate

        Input: 
        xtest: Feature Matrix of test Set\n
        ytest: Value Matrix of test Set\n
        set = test set (default), make it 'train'
        to evaluate performance on train set\n

        Output:
        cost, ypred\n
       '''
        xtest = xtest
        ytest = ytest.reshape(-1,1)
        if (len(xtest)!= len(ytest)):
            raise TinyDropError('Feature and Label matrix have different lenghts')

        self.ypred = self.sigmoid((xtest @ self.w)+ self.b)
        self.cost = ( -1/xtest.shape[0] ) * (np.sum( (ytest * np.log(self.ypred)) + ((1 - ytest) * np.log(1 - self.ypred))))
        return self.cost