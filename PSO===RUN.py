import pandas as pd
from pso_numpy import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

# Load the dataset
df = pd.read_csv('heart.csv')

# Preprocess the dataset
X = df.drop('output', axis=1)
Y = df['output']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the number of nodes in each layer
INPUT_NODES = 13  # Updated to match the number of features in your dataset
HIDDEN_NODES = 100
OUTPUT_NODES = 2

"""
def one_hot_encode(Y):
    num_unique = len(np.unique(np.array(Y)))
    zeros = np.zeros((len(Y), num_unique))
    zeros[range(len(Y)), Y] = 1
    return zeros
"""

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def Negative_Likelihood(probs, Y):
    num_samples = len(probs)
    corect_logprobs = -np.log(probs[range(num_samples), Y])
    return np.sum(corect_logprobs) / num_samples

def Cross_Entropy(probs, Y):
    num_samples = len(probs)
    ind_loss = np.max(-1 * Y * np.log(probs + 1e-12), axis=1)
    return np.sum(ind_loss) / num_samples

def forward_pass(X, Y, W):
    if isinstance(W, Particle):
        W = W.x

    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)

    return Negative_Likelihood(probs, Y)

def predict(X, W):
    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred = np.argmax(probs, axis=1)
    return Y_pred

def get_accuracy(Y, Y_pred):
    return (Y == Y_pred).mean()

if __name__ == "__main__":
    no_solution = 100
    no_dim = (
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    )
    w_range = (0.0, 1.0)
    lr_range = (0.0, 1.0)
    iw_range = (0.9, 0.9)
    c = (0.5, 0.3)

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)

    start_time = time.time()

    s.optimize(forward_pass, X, Y, 100, 1000)

    end_time = time.time()
    time_consumed = end_time - start_time
    print("Time Consumed: %.3f seconds" % time_consumed)

    W = s.get_best_solution()
    Y_pred = predict(X, W)

    accuracy = get_accuracy(Y, Y_pred)
    print("Accuracy: %.3f" % accuracy)

    precision = precision_score(Y, Y_pred, average='macro')
    recall = recall_score(Y, Y_pred, average='macro')
    f1 = f1_score(Y, Y_pred, average='macro')
    auc_score = roc_auc_score(Y, Y_pred)

    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1 Score: %.3f" % f1)
    print("AUC Score: %.3f" % auc_score)

    fpr, tpr, thresholds = roc_curve(Y, Y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()