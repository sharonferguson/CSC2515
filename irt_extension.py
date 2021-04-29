# -*- coding: utf-8 -*-
"""IRT-Extension.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sW6jhPp8EADQsUprenQfiOqtUbI_YyT4
"""

from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tqdm

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred)))  / len(data["is_correct"])

def sigmoid(x, rand_guess=0.25):
    """ Apply sigmoid function - with random guessing taken into account
    """
    return rand_guess + (1-rand_guess)*_sigmoid(x)

def _sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    df = data
    c = df['is_correct']
    y = sigmoid(theta[df['user_id']] - beta[df['question_id']])
    log_lklihood = sum(c*np.log(y) - (1-c)*np.log(1-y))

    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    df = data
    N = len(df)
    
    for i in range(len(theta)):
        subdf = df[df['user_id'] == i] # get all samples where user_id = i 
        c = subdf['is_correct']
        y = sigmoid(theta[i] - beta[subdf['question_id']])
        summation = sum(c*(1-y) - y*(1-c))
        summation = -summation / N
        theta[i] = theta[i] - lr*summation
    
    for j in range(len(beta)):
        subdf = df[df['question_id'] == j] # get all samples where question_id = j
        c = subdf['is_correct']
        y = sigmoid(theta[subdf['user_id']] - beta[j])
        summation = sum(y*(1-c) - c*(1-y))
        summation = -summation / N
        beta[j] = beta[j] - lr*summation
        
    return theta, beta

train_data = load_train_csv("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

train_data = pd.DataFrame(train_data)
val_data = pd.DataFrame(val_data)
test_data = pd.DataFrame(test_data)

ucount = len(np.unique(train_data['user_id']))
qcount = len(np.unique(train_data['question_id']))

"""# ML Training"""

lr = 500
iterations = 50
theta = np.zeros(ucount)
beta = np.zeros(qcount)

neg_log_likelihood(train_data, theta=theta, beta=beta)

# train IRT model
train_nnlks = []
val_nnlks = []
train_acc = []
val_acc = []

prev_train_neg_lld = np.inf
count = 0
for i in range(iterations):
    train_neg_lld = neg_log_likelihood(train_data, theta, beta)
    val_neg_lld = neg_log_likelihood(val_data, theta, beta)
    train_score = evaluate(train_data, theta, beta)
    val_score = evaluate(val_data, theta, beta)
    
    train_acc.append(train_score)
    val_acc.append(val_score)
    train_nnlks.append(train_neg_lld)
    val_nnlks.append(val_neg_lld)
    print("Iteration {} \t Train NLLK: {} \t Val NLLK: {} \t Train Score {} \t Val Score: {}".format(
        i, train_neg_lld, val_neg_lld, train_score, val_score))
    theta, beta = update_theta_beta(train_data, lr, theta, beta)

np.save("theta.npy", theta)
np.save("beta.npy", beta)

theta = np.load("theta.npy")
beta = np.load("beta.npy")

plt.ylabel("Negative Log Likelihood")
plt.xlabel("number of iterations")
plt.plot(train_nnlks)
plt.plot(val_nnlks)
plt.show()

plt.xlabel("number of iterations")
plt.ylabel("Accuracy Score")
plt.plot(val_acc)
plt.plot(train_acc)
plt.show()

print("Final Validation Accuracy: ", evaluate(val_data, theta, beta))
print("Final Testing Accuracy: ", evaluate(test_data, theta, beta))

# PART D
df = pd.DataFrame(train_data)

# select 5 random questions
q_ids = random.sample(range(len(beta)), 5)
for q_id in q_ids:
    thetas = np.arange(-5, 5, 0.1)
    probs = sigmoid(thetas - beta[q_id])
    plt.plot(thetas, probs)

plt.xlabel("Theta - Student's Ability")
plt.ylabel("Probability of the Correct Response")
plt.legend(q_ids)
plt.show()

# EXTRA
theta_beta = np.arange(-5, 5, 0.1)
probs = _sigmoid(theta_beta)
plt.plot(theta_beta, probs)
plt.xlabel(r'$\theta_i - \beta_j$', fontsize=12)
plt.ylabel("Probability of the Correct Response")
plt.show()
probs = sigmoid(theta_beta)
plt.plot(theta_beta, probs)
plt.plot(theta_beta, np.ones(len(theta_beta))*0.25, "--")
# plt.text(3.5, 0.27, 'p=0.25')
plt.ylim(ymin=0)
plt.xlabel(r'$\theta_i - \beta_j$', fontsize=12)
plt.ylabel("Probability of the Correct Response")
plt.show()

