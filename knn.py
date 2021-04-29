from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import pandas as pd
import numpy as np

#used this function to test out different distance metrics
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
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
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
    mat = nbrs.fit_transform(matrix_t)
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


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    train_data = load_train_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    ks = range(1,30)

    #compute validation accuracy for each k and plot as a function of k
    accuracy = []
    for k in ks: 
      acc = knn_impute_by_user(sparse_matrix, val_data, k)
      accuracy.append(acc)

    print(accuracy)
    #plot
    plt.plot(ks, accuracy)
    #plt.title("K-NN alidation accuracy (impute user) as a function of k")
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.show()

    #choose the k with the highest accuracy
    k_star = ks[np.argmax(accuracy)]

    #report the test accuracy using k_star

    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, k_star)

    print("the test accuracy on K* for the impute by user model is: " + str(test_accuracy))

    ##now complete a) and b) for the version "impute by item"

    accuracy_item = []
    for k in ks: 
      acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
      accuracy_item.append(acc_item)

    #plot
    plt.plot(ks, accuracy_item)
    #plt.title("K-NN validation accuracy (impute item) as a function of k")
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.show()

    #choose the k with the highest accuracy

    k_star_item = ks[np.argmax(accuracy_item)]

    #report the test accuracy using k_star

    test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)

    print("the test accuracy on K* for the impute by item model is: " + str(test_accuracy_item))

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()