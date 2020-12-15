import numpy as np

class linear_regression(tmc):
    
    '''
  Linear Regression

  Maps a relationship between given set of variables\n

  Input:
  xtrain: Feature Matrix\n
  Input should be of form: Features along rows
  ytrain: Matrix containing coressponding values\n
  Similarly xtest and ytest\n
  learning_rate\n
  #. of iterations\n

  Various functions:

  train(): To train the model\n
  validate(): To validate the results\n
  r2Score is the metric\n


  Output:
  ypred = Predicted values (Can retrieve after validation)\n
  r2score = performance metric\n
  parameters = Weights\n


  Author:
  Alluri L S V Siddhartha Varma\n


  '''
  
    def __init__(self):
        
        self.param = None
        self.r2score = None
        self.ypred = None

    def train(self, xtrain, ytrain, learning_rate = 0.01, iterations = 100):
        '''
        Train

        Input: 
        xtrain: Feature Matrix of train Set\n
        ytrain: Value Matrix of train Set\n
        Learning rate\n
        No. of iterations\n 

        Output:
        Trained model\n
       '''
        x = xtrain
        y = ytrain.reshape(-1,1)

        if (len(x)!= len(y)):
            raise TinyDropError('Feature and Value matrix have different lenghts')

        learning_rate = learning_rate
        iterations = iterations
        m = xtrain.shape[0]
        n = xtrain.shape[1]
        x = np.hstack((np.ones((m,1)), x))
        self.param = np.zeros((n+1,1))
        for i in range(iterations):
            self.param = self.param - ((learning_rate/m) * (x.T @ (x @ self.param - y ) ) )


    def validate(self, xtest = None, ytest = None,set = 'test'):
        '''
        Validate

        Input: 
        xtest: Feature Matrix of test Set\n
        ytest: Value Matrix of test Set\n
        set = test set (default), make it 'train'
        to evaluate performance on train set\n

        Output:
        r2score, ypred\n
       '''
        xtest = xtest
        ytest = ytest.reshape(-1,1)
        if (len(xtest)!= len(ytest)):
            raise TinyDropError('Feature and Value matrix have different lenghts')
        xtest = np.hstack((np.ones((np.size(xtest, 0),1)), xtest))
        if( set == 'train'):
            self.ypred = x @ self.param
            self.r2score = 1 - (((y - self.ypred)**2).sum() / ((y - y.mean())**2).sum())

        else:
            self.ypred = xtest @ self.param
            self.r2score = 1 - (((ytest - self.ypred)**2).sum() / ((ytest - ytest.mean())**2).sum())
        
        return self.r2score

    
