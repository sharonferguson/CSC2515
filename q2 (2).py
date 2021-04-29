'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import zipfile
import os
from scipy.special import logsumexp
import matplotlib.pyplot as plt



def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = []
    for i in range(0, 10):
        values_i = data.get_digits_by_label(train_data, train_labels, i) #this is all training examples for digit i
        avg_list = np.mean(values_i, axis=0) #averages every row in each column, reminder that rows are examples and each column is a pixel
        means.append(avg_list)

    # Compute means
    return np.array(means)

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    #remember from the assignment instructions we should add 0.01*identity matrix to our answer to avoid numerical errors
    covariances = [] 
    averages = compute_mean_mles(train_data, train_labels)
    for i in range(0, 10):
        values_i = data.get_digits_by_label(train_data, train_labels, i)
        avg_i = averages[i]
        difference_from_mean = values_i - avg_i
        #the covariance matrix - calculated using the formula: 1/m sum(i=1..m) (xi-Mew)(xi-mew).T 
        covariance = np.dot(difference_from_mean.transpose(), difference_from_mean) / values_i.shape[0]

        #add the stabalizing constant
        covariance = covariance + np.eye(values_i.shape[1]) * 0.01

        covariances.append(covariance)
    # Compute covariances
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    #we need to compute this formula: log(p(x|y=k, mew, sigmak)) = log((2*pi)^(-d/2)|sigmak|^(-1/2) exp {-1/2(x-mewk).T sigmak^-1(x-mewk)})
    #will return the log-likelihood of each example being from each category

    likelihoods = [] 
    
    #multiplied terms can now be added once you take the log
    #log(2*pi^-d/2)) becomes (-d/2)*log(2*pi)
    
    first = (-means.shape[0]/2)*np.log(2*np.pi)
    for digit in digits:
        likelihoods_digit = []
        for i in range(0, 10):
            
            #next term is the determinent of sigma to the power of -1/2 which becomes: -1/2 log(det(sigma k))
            second = (-1/2)*np.log(np.linalg.det(covariances[i]))

            #last term becomes -1/2*(x-mewk).T * sigmak^-1(x-mewk)
            inv = np.linalg.inv(covariances[i])
            x_minus_mewk = digit - means[i] 
            #print(inv.shape)
            #print(x_minus_mewk.shape)

            last = (-1/2) * np.matmul(np.transpose(x_minus_mewk),inv)
            last = np.matmul(last, x_minus_mewk)
            likelihoods_digit.append((first+second+last))
        likelihoods.append(likelihoods_digit)
    #print(np.array(likelihoods).shape)
    return np.array(likelihoods)


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    # using bayes rule: p(y|x) = p(x|y)p(y) / P(x) and taking the log of this gives us: log(p(y|x) = log(p(x|y)) + log(p(y)) - log(p(x))
    #in generative_likelihoods we calculated log(p(x|y))

    log_prob_x_y = generative_likelihood(digits, means, covariances)

    log_prob_y = np.log(1/10)

    #p(x) = sum over all y p(x, y) = sum over all y (p(x|y)p(y))

    log_prob_x = logsumexp(log_prob_x_y + log_prob_y, axis=1).reshape(-1, 1)
    
    log_prob_y_x = log_prob_x_y + log_prob_y - log_prob_x

    return log_prob_y_x 

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    #need to get the conditional likelihood for only the correct class labels 
    #slice data where the label is equal to i 
    sums = 0
    for i, likelihoods in enumerate(cond_likelihood):
        sums += likelihoods[int(labels[i])]
    avg = sums/cond_likelihood.shape[0] 
    # Compute as described above and return
    return avg


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    #look at the conditional likelihood and choose the column (out of 10) that has the highest value for each example (row)
    prob_y_given_x = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    #pick the column with the highest likelihood 

    most_likely = np.argmax(prob_y_given_x, axis=1)

    return most_likely

def questiona (train_data, train_labels, test_data, test_labels):

    #compute average conditional log likelihood on both the train and test set 
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    train_avg_conditional_log_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)

    print("training avg log likelihood: " + str(train_avg_conditional_log_likelihood))

    test_avg_conditional_log_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print("test avg log likelihood: " + str(test_avg_conditional_log_likelihood))

def questionb(train_data, train_labels, test_data, test_labels):
    # select the most likely posterior class for each training and test point as your prediction and report your accuracy on the training and test set
    # need to call classify data function and keep track of accuracy 
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    #predict
    train_predictions = classify_data(train_data, means, covariances)
    #create a list of True or False if they were correct
    train_correct = [train_predictions == train_labels]
    #averaging 0s and 1s gives you the percentage of correctly classified data
    train_accuracy = np.mean(train_correct)
    print("train accuracy is: " + str(train_accuracy))

    test_predictions = classify_data(test_data, means, covariances)
    test_correct = [test_predictions == test_labels]
    test_accuracy = np.mean(test_correct)
    print("test accuracy is: " + str(test_accuracy))

def questionc(covariances): 
    # Compute the leading eigenvectors (largest eigenvalue) for each class covariance matrix (can use np.linalg.eig) and plot them side by side as 8 by 8 images.
    
    each_digit = []
    #for each class, compute the eigenvalues and eigenvectors and then choose the eigenvector with the highest associated eigenvalue 
    for i in range(0, 10):
        cov = covariances[i]
        vals, vecs = np.linalg.eig(cov)
        #get largest eigenvalue - this expression is from the np.linalg.eig documentation:
        leading_eigenvector = vecs[:, np.argmax(vals)]
        each_digit.append(leading_eigenvector.reshape(8, 8))
    
    #this only works when run on an interactive python terminal and it seems to be a common issue with plt.imshow 
    plt.title("leading eignvectors")
    plt.imshow(np.concatenate(np.array(each_digit), axis = 1), cmap = 'gray')
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('hw4digits/')

    # Evaluation
    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    #part a
    print("question 2)a)")
    questiona(train_data, train_labels,test_data,test_labels)

    #part b
    print("question 2)b)")
    questionb(train_data, train_labels, test_data, test_labels)

    #part c
    print("question 2)c)")
    questionc(covariances)
 

if __name__ == '__main__':
    main()
