from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import save


# different distance metrics from tutorial

def generate_correct_fraction(base_path):
    """ Load the sparse training data to determine the fraction of correct
        answers on a question-by-question basis.

    :return: frac
        WHERE:
        frac:    fraction of correct answers, shape == (num_questions,)

    """
    
    train_mat = load_train_sparse(base_path).toarray()
    qs_answered = np.isnan(train_mat)
    qs_answered = train_mat.shape[0] - np.sum(qs_answered,axis=0)
    num_correct = np.nansum(train_mat,axis=0)
    frac = num_correct / qs_answered
    
    return frac

def generate_inds(base_path,threshold=50):
    """ Load the sparse training data to determine number of questions answered 
        per student.

    :return: (qs_answered, inds)
        WHERE:
        qs_answered:    number of questions answered per student;
        inds:           indices corresponding to students who answered more 
                        than the threshold number of questions
        (here, the hypothesis is that experience matters, perhaps it is better 
         to train only on those students who answered more or less questions)
    """
    train_mat = load_train_sparse(base_path).toarray()
    qs_answered = np.isnan(train_mat)
    qs_answered = train_mat.shape[1] - np.sum(qs_answered,axis=1)
    inds = np.where(qs_answered>=threshold)
    anti_inds = np.where(qs_answered <= threshold)
    
    return qs_answered, inds, anti_inds

def generate_complexity(base_path):
    """ Load the question metadata in PyTorch Tensor.

    :return: complexity
        WHERE:
        complexity: number of subjects covered in each question
        (here, the hypothesis is that the more multi-conceptual a question is, 
         the more likely it was a more difficult question)
    """
    meta = pd.read_csv(base_path)
    meta.sort_values(by=['question_id'])
    complexity = np.ones([1,len(meta)])
    
    for i in range(len(meta)):
        complexity[0,i] = len(meta['subject_id'][i][1:-1].split(','))
    
    return complexity

def generate_smart(base_path, threshold=50):
    matrix = load_train_sparse(base_path).toarray()
    percent_correct = []
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        row = row[np.logical_not(np.isnan(row))]
        add = np.sum(row)
        #print(add)
        answered = row.shape[0]
        #print(answered)
        percent = add/answered
        percent_correct.append(percent)
    inds = np.where(np.array(percent_correct)>=threshold)
    anti_inds = np.where(np.array(percent_correct)<threshold)

    return percent_correct, inds, anti_inds
    
    



def cdist_multi_numpy(Xa, Xb, missing_values):
    # sqrt(|Xa|^2 + |Xb|^2 - 2 Xa^T Xb)
    Xa2 = (Xa**2).sum(0).reshape((-1, 1))
    Xb2 = (Xb**2).sum(0).reshape((1, -1))
    XaXb = np.dot(Xa.T, Xb)
    return np.sqrt(Xa2+Xb2-2*XaXb)
    

def cdist_multi_bad_memory(Xa, Xb, missing_values):
    Xa = Xa[:, :, np.newaxis]
    Xb = Xb[:, np.newaxis, :]
    return np.sqrt(((Xa-Xb)**2).sum(0))

def cdist_multi_scidist1(Xa, Xb, missing_values):
    return scidist.cdist(Xa.T, Xb.T)  # return shape of cdist is (M, N)

def cdist_multi_scidist2(Xa, Xb, missing_values):
    return scidist.cdist(Xa.T, Xb.T, 'cosine')  # return shape of cdist is (M, N)

def cdist_multi_scidist3(Xa, Xb, missing_values):
    return scidist.cdist(Xa.T, Xb.T, 'chebyshev')  # return shape of cdist is (M, N)

def cdist_single_scidist(Xa, Xb, missing_values):
    return scidist.jaccard(Xa.T, Xb.T)

 #‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule
def sparse_matrix_evaluate_item(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    #print(len(data['is_correct']))
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if cur_user_id >= len(matrix[0]):
            continue
        if matrix[cur_question_id, cur_user_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_question_id, cur_user_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    print(mat.shape)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #the underlying assumption of this function is that if  question A has the same correct and incorrect answers from other users as question B, 
    #(the same students who got question A correct would also get question B correct), question A's correctness for a specific user would match that of that users correctness on question B
    
    nbrs = KNNImputer(n_neighbors=k)

    #in the implementation above, they take the sparse matrix as-is (where students are rows and questions are columns) to predict based on students. If we want to predict based on questions, we should transpose the matrix
    #transpose the matrix so questions are now the samples and users are the features

    matrix_t = matrix.transpose()
    #print("shape of hard sparse matrix transposed is: " + str(matrix_t.shape))
    mat = nbrs.fit_transform(matrix_t)
    #print(mat.shape)
    acc = sparse_matrix_evaluate_item(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


# def main():
#     sparse_matrix = load_train_sparse("./data").toarray()
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#     train_data = load_train_csv("./data")

#     print("Sparse matrix:")
#     print(sparse_matrix)
#     print("Shape of sparse matrix:")
#     print(sparse_matrix.shape)

#     ks = [1, 6, 11, 16, 21, 26]

#     #compute validation accuracy for each k and plot as a function of k
#     accuracy = []
#     for k in ks: 
#       acc = knn_impute_by_user(sparse_matrix, val_data, k)
#       accuracy.append(acc)

#     print(accuracy)
#     # #plot
#     # plt.plot(ks, accuracy)
#     # #plt.title("K-NN alidation accuracy (impute user) as a function of k")
#     # plt.xlabel('k')
#     # plt.ylabel('Validation Accuracy')
#     # plt.show()

#     #choose the k with the highest accuracy
#     for j in range(-10, 11):
#       k_star = ks[np.argmax(accuracy)] +j

#     #report the test accuracy using k_star

#       test_accuracy = knn_impute_by_user(sparse_matrix, test_data, k_star)

#       print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

#     ##now complete a) and b) for the version "impute by item"

#     accuracy_item = []
#     for k in ks: 
#       acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
#       accuracy_item.append(acc_item)

#     # #plot
#     # plt.plot(ks, accuracy_item)
#     # #plt.title("K-NN validation accuracy (impute item) as a function of k")
#     # plt.xlabel('k')
#     # plt.ylabel('Validation Accuracy')
#     # plt.show()

#     #choose the k with the highest accuracy

#     for i in range(-10, 11):

#       k_star_item = ks[np.argmax(accuracy_item)] +i 

#     #report the test accuracy using k_star

#       test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)

#       print("the test accuracy on K* for the impute by item model is: " + str(test_accuracy_item))

#     #####################################################################
#     # TODO:                                                             #
#     # Compute the validation accuracy for each k. Then pick k* with     #
#     # the best performance and report the test accuracy with the        #
#     # chosen k*.                                                        #
#     #####################################################################
#     pass
#     #####################################################################
#     #                       END OF YOUR CODE                            #
#     #####################################################################


#try KNN where there is a split for the questions that are hard vs easy

# def main():
#     sparse_matrix = load_train_sparse("./data").toarray()
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#     train_data = load_train_csv("./data")
    
#     frac = generate_correct_fraction("./data")

#     print(sparse_matrix.shape)
#     hard = np.array([frac<=0.5]).flatten()
#     print(hard.shape)
#     easy = np.array([frac>=0.5]).flatten()
#     print(sum(easy))

#     #map the original index to the new index
#     mapping_h = {}
#     mapping_e = {}
#     count_h = 0
#     count_e = 0
#     for index in range(len(sparse_matrix.T)):
#         if hard[index] == True:
#             mapping_h[index] = count_h
#             count_h +=1
#         else:
#             mapping_e[index]=count_e
#             count_e +=1
            
#     #print(mapping_h)
#     #print(mapping_e)


#     hard_sparse_matrix = sparse_matrix[:,hard]
#     easy_sparse_matrix = sparse_matrix[:,easy]

#     #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))

#     #print(easy_sparse_matrix.shape)
#     #print(val_data)
#     indices_h = [i for i, x in enumerate(hard) if x]
#     indices_e = [i for i, x in enumerate(easy) if x]

#     #split val set
#     val_h = {'question_id': [], 'user_id': [], 'is_correct': []}
#     val_e = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(val_data['question_id'])):
#         if val_data['question_id'][i] in indices_h:
#             val_h['question_id'].append(mapping_h[val_data['question_id'][i]])
#             val_h['user_id'].append(val_data['user_id'][i])
#             val_h['is_correct'].append(val_data['is_correct'][i])
#         else:
#             val_e['question_id'].append(mapping_e[val_data['question_id'][i]])
#             val_e['user_id'].append(val_data['user_id'][i])
#             val_e['is_correct'].append(val_data['is_correct'][i])

#     #split test set

#     test_h = {'question_id': [], 'user_id': [], 'is_correct': []}
#     test_e = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(test_data['question_id'])):
#         if test_data['question_id'][i] in indices_h:
#             test_h['question_id'].append(mapping_h[test_data['question_id'][i]])
#             test_h['user_id'].append(test_data['user_id'][i])
#             test_h['is_correct'].append(test_data['is_correct'][i])
#         else:
#             test_e['question_id'].append(mapping_e[test_data['question_id'][i]])
#             test_e['user_id'].append(test_data['user_id'][i])
#             test_e['is_correct'].append(test_data['is_correct'][i])
        
#     print("shape of easy sparse matrix is: " + str(easy_sparse_matrix.shape))
#   #KNN for hard questions

#     ks = [1, 6, 11, 16, 21, 26]

#     #compute validation accuracy for each k and plot as a function of k
#     accuracy = []
#     for k in ks: 
#         acc = knn_impute_by_user(easy_sparse_matrix, val_e, k)
#         accuracy.append(acc)

#     print(accuracy)
#         # #plot
#         # plt.plot(ks, accuracy)
#         # #plt.title("K-NN alidation accuracy (impute user) as a function of k")
#         # plt.xlabel('k')
#         # plt.ylabel('Validation Accuracy')
#         # plt.show()

#         #choose the k with the highest accuracy
#     for j in range(-10, 11):
#         k_star = ks[np.argmax(accuracy)] +j

#         #report the test accuracy using k_star
#         #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))
#         test_accuracy = knn_impute_by_user(easy_sparse_matrix, test_e, k_star)

#         print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

#         ##now complete a) and b) for the version "impute by item"

#     accuracy_item = []
#     for k in ks: 
#         acc_item = knn_impute_by_item(easy_sparse_matrix, val_e, k)
#         accuracy_item.append(acc_item)

#         # #plot
#         # plt.plot(ks, accuracy_item)
#         # #plt.title("K-NN validation accuracy (impute item) as a function of k")
#         # plt.xlabel('k')
#         # plt.ylabel('Validation Accuracy')
#         # plt.show()

#         #choose the k with the highest accuracy

#     for i in range(-10, 11):

#         k_star_item = ks[np.argmax(accuracy_item)] +i 

#         #report the test accuracy using k_star

#         test_accuracy_item = knn_impute_by_item(easy_sparse_matrix, test_e, k_star_item)

#         print("the test accuracy on K* for the impute by item model is: " + str(test_accuracy_item))


#try KNN where there is a split for students that answered a lot of questions vs few
# def main():
#     sparse_matrix = load_train_sparse("./data").toarray()
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#     train_data = load_train_csv("./data")
#     qs_answered, inds, anti_inds = generate_inds("./data",threshold=148)
#     #print(np.mean(qs_answered))
#     #print(np.quantile(qs_answered, 0.50))
#     #print(np.quantile(qs_answered, 0.75))

#     #map the original index to the new index
#     mapping_many = {}
#     mapping_few = {}
#     count_many = 0
#     count_few = 0
#     for index in range(len(sparse_matrix)):
#         if index in inds[0]:
#             mapping_many[index] = count_many
#             count_many +=1
#         else:
#             mapping_few[index]=count_few
#             count_few +=1
            
#     #print(mapping_many)
#     #print(mapping_few)


#     many_sparse_matrix = sparse_matrix[inds[0],: ]
#     few_sparse_matrix = sparse_matrix[anti_inds[0], :]

#     #split val set
#     val_many = {'question_id': [], 'user_id': [], 'is_correct': []}
#     val_few = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(val_data['user_id'])):
#         if val_data['user_id'][i] in inds[0]:
#             val_many['question_id'].append(val_data['question_id'][i])
#             val_many['user_id'].append(mapping_many[val_data['user_id'][i]])
#             val_many['is_correct'].append(val_data['is_correct'][i])
#         else:
#             val_few['question_id'].append(val_data['question_id'][i])
#             val_few['user_id'].append(mapping_few[val_data['user_id'][i]])
#             val_few['is_correct'].append(val_data['is_correct'][i])

#     print(max(val_many['user_id']))
#     #split test set

#     test_many = {'question_id': [], 'user_id': [], 'is_correct': []}
#     test_few = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(test_data['user_id'])):
#         if test_data['user_id'][i] in inds[0]:
#             test_many['question_id'].append(test_data['question_id'][i])
#             test_many['user_id'].append(mapping_many[test_data['user_id'][i]])
#             test_many['is_correct'].append(test_data['is_correct'][i])
#         else:
#             test_few['question_id'].append(test_data['question_id'][i])
#             test_few['user_id'].append(mapping_few[test_data['user_id'][i]])
#             test_few['is_correct'].append(test_data['is_correct'][i])

#     print(len(val_many['user_id']))
#     print(len(val_many['is_correct']))
        
#     print("shape of easy sparse matrix is: " + str(few_sparse_matrix.shape))

# #only run knn by user here

#     ks = [1, 6, 11, 16, 21, 26]

#     #compute validation accuracy for each k and plot as a function of k
#     accuracy = []
#     for k in ks: 
#         acc = knn_impute_by_user(many_sparse_matrix, val_many, k)
#         accuracy.append(acc)

#     print(accuracy)
#         # #plot
#         # plt.plot(ks, accuracy)
#         # #plt.title("K-NN alidation accuracy (impute user) as a function of k")
#         # plt.xlabel('k')
#         # plt.ylabel('Validation Accuracy')
#         # plt.show()

#         #choose the k with the highest accuracy
#     for j in range(0, 11):
#         k_star = ks[np.argmax(accuracy)] +j

#         #report the test accuracy using k_star
#         #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))
#         test_accuracy = knn_impute_by_user(many_sparse_matrix, test_many, k_star)

#         print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

 
# 
 #try KNN where there is a split for complex vs simple questions  

# def main():
#     sparse_matrix = load_train_sparse("./data").toarray()
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#     train_data = load_train_csv("./data")
    
#     complexity = generate_complexity("data/question_meta.csv")

    

#     #print(sparse_matrix.shape)
#     complexx = np.array([complexity >4]).flatten()
#     #print(hard.shape)
#     simple = np.array([complexity <= 4]).flatten()
#     #print(sum(easy))

#     #map the original index to the new index
#     mapping_c = {}
#     mapping_s = {}
#     count_c = 0
#     count_s = 0
#     for index in range(len(sparse_matrix.T)):
#         if complexx[index] == True:
#             mapping_c[index] = count_c
#             count_c +=1
#         else:
#             mapping_s[index]=count_s
#             count_s +=1
            
#     #print(mapping_h)
#     #print(mapping_e)


#     c_sparse_matrix = sparse_matrix[:,complexx]
#     s_sparse_matrix = sparse_matrix[:,simple]

#     #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))

#     #print(easy_sparse_matrix.shape)
#     #print(val_data)
#     indices_c = [i for i, x in enumerate(complexx) if x]
#     indices_s = [i for i, x in enumerate(simple) if x]

#     #split val set
#     val_c = {'question_id': [], 'user_id': [], 'is_correct': []}
#     val_s = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(val_data['question_id'])):
#         if val_data['question_id'][i] in indices_c:
#             val_c['question_id'].append(mapping_c[val_data['question_id'][i]])
#             val_c['user_id'].append(val_data['user_id'][i])
#             val_c['is_correct'].append(val_data['is_correct'][i])
#         else:
#             val_s['question_id'].append(mapping_s[val_data['question_id'][i]])
#             val_s['user_id'].append(val_data['user_id'][i])
#             val_s['is_correct'].append(val_data['is_correct'][i])

#     #split test set

#     test_c = {'question_id': [], 'user_id': [], 'is_correct': []}
#     test_s = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(test_data['question_id'])):
#         if test_data['question_id'][i] in indices_c:
#             test_c['question_id'].append(mapping_c[test_data['question_id'][i]])
#             test_c['user_id'].append(test_data['user_id'][i])
#             test_c['is_correct'].append(test_data['is_correct'][i])
#         else:
#             test_s['question_id'].append(mapping_s[test_data['question_id'][i]])
#             test_s['user_id'].append(test_data['user_id'][i])
#             test_s['is_correct'].append(test_data['is_correct'][i])
        
#     print("shape of easy sparse matrix is: " + str(s_sparse_matrix.shape))
#   #KNN for hard questions

#     ks = [1, 6, 11, 16, 21, 26]

# #     #compute validation accuracy for each k and plot as a function of k
# #     accuracy = []
# #     for k in ks: 
# #         acc = knn_impute_by_user(easy_sparse_matrix, val_e, k)
# #         accuracy.append(acc)

# #     print(accuracy)
# #         # #plot
# #         # plt.plot(ks, accuracy)
# #         # #plt.title("K-NN alidation accuracy (impute user) as a function of k")
# #         # plt.xlabel('k')
# #         # plt.ylabel('Validation Accuracy')
# #         # plt.show()

# #         #choose the k with the highest accuracy
# #     for j in range(-10, 11):
# #         k_star = ks[np.argmax(accuracy)] +j

# #         #report the test accuracy using k_star
# #         #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))
# #         test_accuracy = knn_impute_by_user(easy_sparse_matrix, test_e, k_star)

# #         print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

# #         ##now complete a) and b) for the version "impute by item"

#     accuracy_item = []
#     for k in ks: 
#         acc_item = knn_impute_by_item(c_sparse_matrix, val_c, k)
#         accuracy_item.append(acc_item)

#         # #plot
#         # plt.plot(ks, accuracy_item)
#         # #plt.title("K-NN validation accuracy (impute item) as a function of k")
#         # plt.xlabel('k')
#         # plt.ylabel('Validation Accuracy')
#         # plt.show()

#         #choose the k with the highest accuracy

#     for i in range(0, 11):

#         k_star_item = ks[np.argmax(accuracy_item)] +i 

#         #report the test accuracy using k_star

#         test_accuracy_item = knn_impute_by_item(c_sparse_matrix, test_c, k_star_item)

#         print("the test accuracy on K* for the impute by item model is: " + str(test_accuracy_item))
    
  #try KNN where there is a split for smart vs dumb students
# def main():
#     sparse_matrix = load_train_sparse("./data").toarray()
#     val_data = load_valid_csv("./data")
#     test_data = load_public_test_csv("./data")
#     train_data = load_train_csv("./data")
#     percent_correct, inds, anti_inds = generate_smart("./data", threshold=0.78)
#     print(np.mean(percent_correct))
#     print(np.quantile(percent_correct, 0.50))
#     print(np.quantile(percent_correct, 0.75))

#     #map the original index to the new index
#     mapping_smart = {}
#     mapping_dumb = {}
#     count_smart = 0
#     count_dumb = 0
#     for index in range(len(sparse_matrix)):
#         if index in inds[0]:
#             mapping_smart[index] = count_smart
#             count_smart +=1
#         else:
#             mapping_dumb[index]=count_dumb
#             count_dumb +=1
            
#     #print(mapping_many)
#     #print(mapping_few)


#     smart_sparse_matrix = sparse_matrix[inds[0],: ]
#     dumb_sparse_matrix = sparse_matrix[anti_inds[0], :]

#     #split val set
#     val_smart = {'question_id': [], 'user_id': [], 'is_correct': []}
#     val_dumb = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(val_data['user_id'])):
#         if val_data['user_id'][i] in inds[0]:
#             val_smart['question_id'].append(val_data['question_id'][i])
#             val_smart['user_id'].append(mapping_smart[val_data['user_id'][i]])
#             val_smart['is_correct'].append(val_data['is_correct'][i])
#         else:
#             val_dumb['question_id'].append(val_data['question_id'][i])
#             val_dumb['user_id'].append(mapping_dumb[val_data['user_id'][i]])
#             val_dumb['is_correct'].append(val_data['is_correct'][i])

#     print(max(val_smart['user_id']))
#     #split test set

#     test_smart = {'question_id': [], 'user_id': [], 'is_correct': []}
#     test_dumb = {'question_id': [], 'user_id': [], 'is_correct': []}

#     for i in range(len(test_data['user_id'])):
#         if test_data['user_id'][i] in inds[0]:
#             test_smart['question_id'].append(test_data['question_id'][i])
#             test_smart['user_id'].append(mapping_smart[test_data['user_id'][i]])
#             test_smart['is_correct'].append(test_data['is_correct'][i])
#         else:
#             test_dumb['question_id'].append(test_data['question_id'][i])
#             test_dumb['user_id'].append(mapping_dumb[test_data['user_id'][i]])
#             test_dumb['is_correct'].append(test_data['is_correct'][i])

#     print("the number of smart users are" + str(len(test_smart['user_id'])))
#     print("the number of smart users are" + str(len(test_dumb['user_id'])))
    
        
#     print("shape of easy sparse matrix is: " + str(dumb_sparse_matrix.shape))

# #only run knn by user here

#     ks = [1, 6, 11, 16, 21, 26]

#     #compute validation accuracy for each k and plot as a function of k
#     accuracy = []
#     for k in ks: 
#         acc = knn_impute_by_user(smart_sparse_matrix, val_smart, k)
#         accuracy.append(acc)

#     print(accuracy)
#         # #plot
#         # plt.plot(ks, accuracy)
#         # #plt.title("K-NN alidation accuracy (impute user) as a function of k")
#         # plt.xlabel('k')
#         # plt.ylabel('Validation Accuracy')
#         # plt.show()

#         #choose the k with the highest accuracy
#     for j in range(0, 11):
#         k_star = ks[np.argmax(accuracy)] +j

#         #report the test accuracy using k_star
#         #print("shape of hard sparse matrix is: " + str(hard_sparse_matrix.shape))
#         test_accuracy = knn_impute_by_user(smart_sparse_matrix, test_smart, k_star)

#         print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

def main():
    percent_correct, inds, anti_inds = generate_smart("./data", threshold=0.56)
    sparse_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    train_data = np.array([train_data['question_id'], train_data['user_id'], train_data['is_correct']])
    print(train_data)

    d = []
    for i in range(len(sparse_matrix.T)):
        avg_smart = np.nanmean(sparse_matrix[inds,i])
        #print(avg_smart)
        avg_dumb = np.nanmean(sparse_matrix[anti_inds, i])
        #print(avg_dumb)
        discrim = np.abs(avg_smart - avg_dumb)
        #print(discrim)
        d.append(discrim)
        #print(d)

    d_max = np.nanmax(d)
    print(d_max)
    d = d/d_max
    print(d) 

    save('discrim.npy', d)

    plt.plot(range(1,1775), d)
    plt.show()
        
  


  


if __name__ == "__main__":
    main()