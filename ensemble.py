import numpy as np 
import pandas as pd 
from utils import *
from nn import *
from knn import *
import scipy.sparse as sp
from item_response import *


def knn_user_ensemble(train_data, val_data, test_data, validation_accuracy, test_accuracy):
    '''ensemble of 3 user-based collaborative filtering k-NN models using the top 3 k values'''
    for i in range(50):  
        sample_1 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset

        sparse_sample_1 = np.empty((542,1774))
        sparse_sample_1[:] = np.NaN

        for i in range(len(sample_1)):
            sparse_sample_1[sample_1[i][1], sample_1[i][0]] = sample_1[i][2]
        
        #print(sparse_sample)
        #get KNN model - best model was impute_user with k=11

        nbrs1 = KNNImputer(n_neighbors=7)
        mat1 = nbrs1.fit_transform(sparse_sample_1) #this now contains all of the predictions, just needs to be indexed

        sample_2 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset

        sparse_sample_2 = np.empty((542,1774))
        sparse_sample_2[:] = np.NaN

        for i in range(len(sample_2)):
            sparse_sample_2[sample_2[i][1], sample_2[i][0]] = sample_2[i][2]

        #2nd best KNN
        nbrs2 = KNNImputer(n_neighbors=9)
        mat2 = nbrs2.fit_transform(sparse_sample_2) #this now contains all of the predictions, just needs to be indexed

        sample_3 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset

        sparse_sample_3 = np.empty((542,1774))
        sparse_sample_3[:] = np.NaN

        for i in range(len(sample_3)):
            sparse_sample_3[sample_3[i][1], sample_3[i][0]] = sample_3[i][2]

        #3rd best KNN
        nbrs3 = KNNImputer(n_neighbors=11)
        mat3 = nbrs3.fit_transform(sparse_sample_3) #this now contains all of the predictions, just needs to be indexed

        #now go through each value in the validation data, get the prediction from all 3 models, sum, if its >=2, then predict 1, if its < 2 predict 0.

        total_accurate = 0
        total_prediction=0
        for i in range(len(val_data["is_correct"])):
            cur_user_id = val_data["user_id"][i]
            cur_question_id = val_data["question_id"][i]
            # if cur_question_id >= len(mat1[:, 0]):
            #     continue
            if mat1[cur_user_id, cur_question_id] >= 0.5:
                vote_1 =1
            if mat1[cur_user_id, cur_question_id] < 0.5:
                vote_1 = 0
            if mat2[cur_user_id, cur_question_id] >= 0.5:
                vote_2 =1
            if mat2[cur_user_id, cur_question_id] < 0.5:
                vote_2 = 0
            if mat3[cur_user_id, cur_question_id] >= 0.5:
                vote_3 =1
            if mat3[cur_user_id, cur_question_id] < 0.5:
                vote_3 = 0 
            total = vote_1 + vote_2 + vote_3
            if total >= 2:
                prediction = 1
            else:
                prediction = 0
            if prediction ==1 and val_data["is_correct"][i]:
                total_accurate += 1
            if prediction ==0 and not val_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        
        validation_accuracy.append(total_accurate / float(total_prediction))

        total_accurate = 0
        total_prediction=0
        for i in range(len(test_data["is_correct"])):
            cur_user_id = test_data["user_id"][i]
            cur_question_id = test_data["question_id"][i]
            # if cur_question_id >= len(matrix[:, 0]):
            #     continue
            if mat1[cur_user_id, cur_question_id] >= 0.5:
                vote_1 =1
            if mat1[cur_user_id, cur_question_id] < 0.5:
                vote_1 = 0
            if mat2[cur_user_id, cur_question_id] >= 0.5:
                vote_2 =1
            if mat2[cur_user_id, cur_question_id] < 0.5:
                vote_2 = 0
            if mat3[cur_user_id, cur_question_id] >= 0.5:
                vote_3 =1
            if mat3[cur_user_id, cur_question_id] < 0.5:
                vote_3 = 0 
            total = vote_1 + vote_2 + vote_3
            if total >= 2:
                prediction = 1
            else:
                prediction = 0
            if prediction ==1 and test_data["is_correct"][i]:
                total_accurate += 1
            if prediction ==0 and not test_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        test_accuracy.append(total_accurate / float(total_prediction))


    plt.plot(range(1,51), validation_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.plot(range(1,51), test_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Test Accuracy")
    plt.show()

    print("the average validation accuracy is: " + str(np.mean(validation_accuracy)))
    print("the average test accuracy is: " + str(np.mean(test_accuracy)))
    print("the stdev validation accuracy is: " + str(np.std(validation_accuracy)))
    print("the stdev test accuracy is: " + str(np.std(test_accuracy)))

def knn_item_ensemble(train_data, val_data, test_data, validation_accuracy, test_accuracy): 
    '''ensemble of 3 item-based collaborative filtering k-NN models using the top 3 k values'''
    for i in range(50):  
        sample_1 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset
        sample_1_t = sample_1.transpose()

        sparse_sample_1 = np.empty((542,1774))
        sparse_sample_1[:] = np.NaN

        for i in range(len(sample_1)):
            sparse_sample_1[sample_1[i][1], sample_1[i][0]] = sample_1[i][2]

        sparse_sample_1_t = sparse_sample_1.transpose()
        
        #print(sparse_sample)
        #get KNN model - best model was impute_user with k=11

        nbrs1 = KNNImputer(n_neighbors=29)
        mat1 = nbrs1.fit_transform(sparse_sample_1_t) #this now contains all of the predictions, just needs to be indexed

        sample_2 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset
        sample_2_t = sample_2.transpose()

        sparse_sample_2 = np.empty((542,1774))
        sparse_sample_2[:] = np.NaN

        for i in range(len(sample_2)):
            sparse_sample_2[sample_2[i][1], sample_2[i][0]] = sample_2[i][2]

        sparse_sample_2_t = sparse_sample_2.transpose()
        #2nd best KNN
        nbrs2 = KNNImputer(n_neighbors=25)
        mat2 = nbrs2.fit_transform(sparse_sample_2_t) #this now contains all of the predictions, just needs to be indexed

        sample_3 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset
        sample_3_t = sample_3.transpose()

        sparse_sample_3 = np.empty((542,1774))
        sparse_sample_3[:] = np.NaN

        for i in range(len(sample_3)):
            sparse_sample_3[sample_3[i][1], sample_3[i][0]] = sample_3[i][2]

        sparse_sample_3_t = sparse_sample_3.transpose()
        #3rd best KNN
        nbrs3 = KNNImputer(n_neighbors=21)
        mat3 = nbrs3.fit_transform(sparse_sample_3_t) #this now contains all of the predictions, just needs to be indexed

        #now go through each value in the validation data, get the prediction from all 3 models, sum, if its >=2, then predict 1, if its < 2 predict 0.

        total_accurate = 0
        total_prediction=0
        for i in range(len(val_data["is_correct"])):
            cur_user_id = val_data["user_id"][i]
            cur_question_id = val_data["question_id"][i]
            # if cur_question_id >= len(matrix[:, 0]):
            #     continue
            if mat1[cur_question_id, cur_user_id] >= 0.5:
                vote_1 =1
            if mat1[cur_question_id, cur_user_id] < 0.5:
                vote_1 = 0
            if mat2[cur_question_id, cur_user_id] >= 0.5:
                vote_2 =1
            if mat2[cur_question_id, cur_user_id] < 0.5:
                vote_2 = 0
            if mat3[cur_question_id, cur_user_id] >= 0.5:
                vote_3 =1
            if mat3[cur_question_id, cur_user_id] < 0.5:
                vote_3 = 0 
            total = vote_1 + vote_2 + vote_3
            if total >= 2:
                prediction = 1
            else:
                prediction = 0
            if prediction ==1 and val_data["is_correct"][i]:
                total_accurate += 1
            if prediction ==0 and not val_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        
        validation_accuracy.append(total_accurate / float(total_prediction))

        total_accurate = 0
        total_prediction=0
        for i in range(len(test_data["is_correct"])):
            cur_user_id = test_data["user_id"][i]
            cur_question_id = test_data["question_id"][i]
            # if cur_question_id >= len(matrix[:, 0]):
            #     continue
            if mat1[cur_question_id, cur_user_id] >= 0.5:
                vote_1 =1
            if mat1[cur_question_id, cur_user_id] < 0.5:
                vote_1 = 0
            if mat2[cur_question_id, cur_user_id] >= 0.5:
                vote_2 =1
            if mat2[cur_question_id, cur_user_id] < 0.5:
                vote_2 = 0
            if mat3[cur_question_id, cur_user_id] >= 0.5:
                vote_3 =1
            if mat3[cur_question_id, cur_user_id] < 0.5:
                vote_3 = 0 
            total = vote_1 + vote_2 + vote_3
            if total >= 2:
                prediction = 1
            else:
                prediction = 0
            if prediction ==1 and test_data["is_correct"][i]:
                total_accurate += 1
            if prediction ==0 and not test_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        test_accuracy.append(total_accurate / float(total_prediction))


    plt.plot(range(1,51), validation_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.plot(range(1,51), test_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Test Accuracy")
    plt.show()

    print("the average validation accuracy is: " + str(np.mean(validation_accuracy)))
    print("the average test accuracy is: " + str(np.mean(test_accuracy)))
    print("the stdev validation accuracy is: " + str(np.std(validation_accuracy)))
    print("the stdev test accuracy is: " + str(np.std(test_accuracy)))


#ensemble with IRT, KNN and NN

def ensemble(train_data, test_data,val_data, validation_accuracy, test_accuracy):
    '''ensemble method using a NN, IRT and user-based K-NN''' 
    for i in range(50):  
        sample_1 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #create the sparse matrix for this subset
        sample_1_t = sample_1.transpose()

        sparse_sample_1 = np.empty((542,1774))
        sparse_sample_1[:] = np.NaN

        for i in range(len(sample_1)):
            sparse_sample_1[sample_1[i][1], sample_1[i][0]] = sample_1[i][2]

        #print(sparse_sample)
        #get KNN model - best model was impute_user with k=11

        nbrs1 = KNNImputer(n_neighbors=11)
        mat1 = nbrs1.fit_transform(sparse_sample_1)

        #create the second dataset 

        sample_2 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        #turn this back into a dictionary 
        sample_2 = sample_2.transpose()

        sample_2_dict = {'question_id': sample_2[0], 'user_id': sample_2[1], 'is_correct': sample_2[2]}

        #train IRT
        lr = 500
        iterations =100
        theta, beta, val_acc_lst = irt(sample_2_dict, val_data, lr, iterations)

        #get third dataset
        sample_3 = train_data[np.random.choice(train_data.shape[0], train_data.shape[0], replace=True), :]

        sparse_sample_3 = np.empty((542,1774))
        sparse_sample_3[:] = np.NaN

        for i in range(len(sample_3)):
            sparse_sample_3[sample_3[i][1], sample_3[i][0]] = sample_3[i][2]

        zero_train_matrix = sparse_sample_3.copy()
        zero_train_matrix[np.isnan(sparse_sample_3)] = 0
        # Change to Float Tensor for PyTorch.
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(sparse_sample_3)

        #train NN
        k = 50
        model = AutoEncoder(train_matrix.shape[1],k)

        # Set optimization hyperparameters.
        lr = 0.01
        num_epoch = 50
        lamb = 0.001

        train(model, lr, lamb, train_matrix, zero_train_matrix,
            val_data, num_epoch)

        #predict for validation set

        pred2 = evaluate_ensemble(val_data, theta, beta)
        pred3, accuracy = evaluate_ensemble_2(model, train_matrix, val_data)

        total_accurate = 0
        total_prediction = 0

        for i in range(len(val_data["is_correct"])):
            cur_user_id = val_data["user_id"][i]
            cur_question_id = val_data["question_id"][i]
            # if cur_question_id >= len(matrix[:, 0]):
            #     continue
            if mat1[cur_user_id, cur_question_id] >= 0.5:
                vote_1 =1
            if mat1[cur_user_id, cur_question_id] < 0.5:
                vote_1 = 0
            vote_2 = pred2[i]
            vote_3 = pred3[i]

            total = vote_1 + vote_2 + vote_3

            if total >= 2:
                final = 1
            else:
                final = 0
            
            if final ==1 and val_data["is_correct"][i]:
                total_accurate += 1
            if final ==0 and not val_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        
        validation_accuracy.append(total_accurate / float(total_prediction))

        #predict for test set

        pred2 = evaluate_ensemble(test_data, theta, beta)
        pred3, accuracy = evaluate_ensemble_2(model, train_matrix, test_data)

        total_accurate = 0
        total_prediction = 0

        for i in range(len(test_data["is_correct"])):
            cur_user_id = test_data["user_id"][i]
            cur_question_id = test_data["question_id"][i]
            # if cur_question_id >= len(matrix[:, 0]):
            #     continue
            if mat1[cur_user_id, cur_question_id] >= 0.5:
                vote_1 =1
            if mat1[cur_user_id, cur_question_id] < 0.5:
                vote_1 = 0
            vote_2 = pred2[i]
            vote_3 = pred3[i]

            total = vote_1 + vote_2 + vote_3

            if total >= 2:
                final = 1
            else:
                final = 0
            
            if final ==1 and test_data["is_correct"][i]:
                total_accurate += 1
            if final ==0 and not test_data["is_correct"][i]:
                total_accurate += 1
            total_prediction +=1
        
        test_accuracy.append(total_accurate / float(total_prediction))

    plt.plot(range(1,51), validation_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Validation Accuracy")
    plt.show()

    plt.plot(range(1,51), test_accuracy)
    plt.xlabel("Bootsrapping Sample")
    plt.ylabel("Test Accuracy")
    plt.show()

    print("the average validation accuracy is: " + str(np.mean(validation_accuracy)))
    print("the average test accuracy is: " + str(np.mean(test_accuracy)))
    print("the stdev validation accuracy is: " + str(np.std(validation_accuracy)))
    print("the stdev test accuracy is: " + str(np.std(test_accuracy)))

def main():
    np.random.seed(123)

    #bootstrap the training dataset into 50 different version. For each version, train 3 different on slightly different bootstrapped samples models
    #then get each model to predict, and then average the predictions

    #read in the datasets

    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    train_data = load_train_csv("./data")

    size = len(train_data['is_correct'])

    train_data = np.array([train_data['question_id'], train_data['user_id'], train_data['is_correct']]).transpose()

    validation_accuracy = []
    test_accuracy = []

    #knn-user
    #knn_user_ensemble(train_data, val_data, test_data, validation_accuracy, test_accuracy)
    #knn-item
    #knn_item_ensemble(train_data, val_data, test_data, validation_accuracy, test_accuracy)
    #ensemble of all methods
    ensemble(train_data, test_data,val_data, validation_accuracy, test_accuracy)

    
if __name__ == "__main__":
    main()