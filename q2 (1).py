from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)
import scipy.special as special

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#print(y)
#print(x)

#print(x.shape)
#print(y.shape)

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
  #we don't have to loop through the test data here because we only input one row at a time
  #calculate the distance based weights

    
    num = np.exp(np.divide(-(l2(np.reshape(test_datum, (1, test_datum.shape[0])), x_train)), 2*(tau**2))) 
    denom = np.exp(special.logsumexp(np.divide(-(l2(np.reshape(test_datum, (1, test_datum.shape[0])), x_train)), 2*(tau**2))))

    a_matrix = np.divide(num, denom)
    #print(a_matrix)

    #create the diagonal matrix
    a_diagonal = np.diag(a_matrix[0,:])
    #print(a_diagonal)


    #solve for w*

    #atax = np.dot(np.dot(x_train.transpose(), a_matrix), x_train)

    atax = np.dot((x_train.transpose()*a_matrix), x_train)
    #atay = np.dot(np.dot(x_train.transpose(), a_matrix), y_train)
    atay = np.dot((x_train.transpose()* a_matrix),  y_train)
    lamI = np.zeros_like(atax)
    np.fill_diagonal(lamI, lam)

    w_star = np.linalg.solve((atax + lamI), atay)

    yhat = np.dot(test_datum, w_star)

    return yhat


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    #split into training and testing 

    #shuffle
    np.random.seed(0)
    idx = np.random.permutation(x.shape[0])
    x_new = x[idx, :]
    y_new = y[idx]

    x_train = x_new[:int(x.shape[0]*(1-val_frac))]
    y_train = y_new[:int(x.shape[0]*(1-val_frac))]
    x_test = x_new[int(x.shape[0]*(1-val_frac)):]
    y_test = y_new[int(x.shape[0]*(1-val_frac)):]
    
    

    #create empty numpy arrays

    train_loss = np.empty(len(taus))
    test_loss = np.empty(len(taus))

    
    #loop over each tau, predict and calculate the loss
    count = 0
    for tau in taus:
        yhat_train = np.empty(x_train.shape[0])
        for i in range(len(x_train)):
            yhat_train[i] = LRLS(x_train[i], x_train, y_train, tau)

        yhat_test = np.empty(x_test.shape[0])
        for j in range(len(x_test)):
          yhat_test[j] = LRLS(x_test[j], x_train, y_train, tau)

    #calculate error

        trainerror = (yhat_train - y_train)
        testerror = (yhat_test - y_test)

    #use mean squared error to summarize this array

        train_loss[count] = np.mean(trainerror**2)
        test_loss[count] = np.mean(testerror**2)

        print(tau)
        print(count, tau, train_loss[count], test_loss[count])

        count += 1 
    return train_loss, test_loss
    ## TODO

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,50)
    #taus = [np.log(10)]
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses)
    plt.xlabel("Tau")
    plt.ylabel("Average Training Loss")
    plt.show()
    plt.semilogx(taus, test_losses)
    plt.xlabel("Tau")
    plt.ylabel("Average Test Loss")
    plt.show()

   


