import numpy as np
from sklearn import cross_validation, svm
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import argparse

parser = argparse.ArgumentParser("""Code to use cross-validation to assess the performance of kernel svms on pairwise-comparisons
""")
parser.add_argument('--phn_set1',type=str,nargs='+',help='first set of phones')
parser.add_argument('--phn_set2',type=str,nargs='+',help='second set of phones')
parser.add_argument('--data_path',type=str,help='path to where the phone sets are stored',default='/var/tmp/stoehr/phoneclassification/data')
parser.add_argument('--save_prefix',type=str,help='prefix for saving the accuracy and results table')
# parser.add_argument('--',type=,help='')
args = parser.parse_args()

def load_data_set(phn_set1,phn_set2, data_path, data_set):
    for phn_id,phn in enumerate(phn_set1):
        if phn_id == 0:
            X_data = np.load('%s/%s_%s_examples_S.npy' % (data_path,
                                                                   phn,data_set)
                                                                   )

            X_shape = X_data.shape[1:]
            X_data = X_data.reshape(len(X_data),
                                     np.prod(X_shape))
            n_data = len(X_data)
            y_data = np.zeros(n_data,dtype=np.intc)
        else:
            X = np.load('%s/%s_%s_examples_S.npy' % (data_path,
                                                                   phn,data_set)
                                                                   )

            X = X.reshape(X.shape[0],np.prod(X_shape))
            while X.shape[0] + n_data > X_data.shape[0]:
                new_X_data = np.zeros((2*X_data.shape[0],X_data.shape[1]))
                new_X_data[:n_data] = X_data[:n_data]
                X_data = new_X_data
                new_y_data = np.zeros(2*len(y_data),dtype=np.intc)
                new_y_data[:n_data] = y_data[:n_data]
                y_data = new_y_data

            X_data[n_data:
                   n_data + len(X)] = X
            y_data[n_data:
                   n_data + len(X)] = 0
            n_data += len(X)

    for phn_id,phn in enumerate(phn_set2):

        X = np.load('%s/%s_%s_examples_S.npy' % (data_path,
                                                                   phn,data_set)
                                                                   )

        X = X.reshape(X.shape[0],np.prod(X_shape))
        while X.shape[0] + n_data > X_data.shape[0]:
            new_X_data = np.zeros((2*X_data.shape[0],X_data.shape[1]))
            new_X_data[:n_data] = X_data[:n_data]
            X_data = new_X_data
            new_y_data = np.zeros(2*len(y_data),dtype=np.intc)
            new_y_data[:n_data] = y_data[:n_data]
            y_data = new_y_data

        X_data[n_data:
                n_data + len(X)] = X
        y_data[n_data:
                n_data + len(X)] = 1
        n_data += len(X)

    return  X_data[:n_data],y_data[:n_data]


X_train, y_train = load_data_set(args.phn_set1,args.phn_set2, args.data_path, 'train')
X_dev, y_dev = load_data_set(args.phn_set1,args.phn_set2, args.data_path, 'dev')

tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-2,1e-3,1e-4,1e-5,1e-1],
                     'C':[.1,1,10,.01,100]}]


print 'commencing training'
error_values = []
for gamma_id, gamma in enumerate([1e-4]):
    for C_id, C in enumerate([100]):
        clf = SVC(C=C,gamma=gamma,kernel='rbf',tol=0.00001,verbose=True)
        clf.fit(X_train,y_train)
        s = clf.score(X_dev,y_dev)
        print "C=%g\tgamma=%g\tscore=%g" % (C,gamma,s)
        error_values.append((gamma,C,s))
        if gamma_id == 0 and C_id ==0:
            best_score = s
            print "updated best score to %g" % best_score
            best_C = C
            best_gamma = gamma

        elif s > best_score:
            best_score = s
            print "updated best score to %g" % best_score
            best_C = C
            best_gamma = gamma



X = np.zeros((len(X_train) + len(X_dev),X_train.shape[1]),dtype=float)
y = np.zeros(len(X_train) + len(X_dev),dtype=int)
X[:len(X_train)] = X_train
X[len(X_train):] = X_dev
y[:len(y_train)] = y_train
y[len(y_train):] = y_dev
clf = SVC(C=best_C,gamma=best_gamma,kernel='rbf',tol=0.00001,verbose=True)
clf.fit(X,y)
X_test, y_test = load_data_set(args.phn_set1,args.phn_set2, args.data_path, 'core_test')
s = clf.score(X_test,
                   y_test)
open('%s_accuracy.txt' % args.save_prefix,'w').write(str(s))
print args.phn_set1,args.phn_set2, s

error_values = np.array(error_values)
np.save('%s_error_values.npy' % args.save_prefix,error_values)
