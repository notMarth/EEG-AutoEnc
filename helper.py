#HELPER FUNCTIONS
#================

from sklearn.preprocessing import StandardScaler

def split_events(X, Y, events, sample_rate, bound):

    new_X = []
    new_Y = []

    for event in events:
        new_X.append(X[:,event - (sample_rate*bound):event + (sample_rate*bound)])
        new_Y.append(Y[:,event - (sample_rate*bound):event + (sample_rate*bound)])

    return new_X, new_Y



#Build regression matrix
#Adapted from https://stackoverflow.com/questions/5842903/block-tridiagonal-matrix-python 
from scipy.sparse import diags
import numpy as np
fs = 250

#lag_mat takes a given input (stimulus) and generates a time-lagged matrix
#currently set up only for the forward model
def lag_mat(stimulus, sample_rate):

    #sampling frequency
    fs = sample_rate
    #start and end in seconds * frequency = num of samples
    start = int(np.floor(-0.25*fs))
    end = int(np.ceil(0.85*fs))

    #time lag list - sample points for the time lags
    lags = list(range(int(np.floor(-0.25 * fs)), int(np.ceil(0.85 * fs)) + 1))
    n_lags = len(lags)

    #Adapted from https://github.com/powerfulbean/mTRFpy/blob/master/mtrf/matrices.py
    x = np.array([stimulus]).T
    n_samples, n_variables = x.shape
    if max(lags) > n_samples:
        raise ValueError("The maximum lag can't be longer than the signal!")
    lag_matrix = np.zeros((n_samples, n_variables * n_lags))

    for idx, lag in enumerate(lags):
        col_slice = slice(idx * n_variables, (idx + 1) * n_variables)
        if lag < 0:
            lag_matrix[0 : n_samples + lag, col_slice] = x[-lag:, :]
        elif lag > 0:
            lag_matrix[lag:n_samples, col_slice] = x[0 : n_samples - lag, :]
        else:
            lag_matrix[:, col_slice] = x


    return lag_matrix


#generate mask for empty portions of data at end of songs
#stimulus should be given as 1-D
#threshold gives the maximum amplitude of the audio envelope to be considered as
#"no audio"
#minimum gives the number of sample points to be under this threshold for the
#current section of the song to be considered the end
def mask(stimulus, threshold, minimum):
    n_samples = len(stimulus)
    song_mask = np.ones(n_samples)
    min_num = minimum
    thresh = threshold
    zeros = 0
    num=0
    
    for sample in range(n_samples):
        if np.abs(stimulus[sample]) <= thresh:
            zeros += 1
            if zeros == min_num:
                song_mask[sample-min_num-1:n_samples] = np.zeros(n_samples - (sample - min_num-1))
                zeros=0
                num+=1
                break
        else:
            zeros = 0


    return song_mask


from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from scipy.sparse import diags
import random

#split input data into epochs
#give to function as 2d array at the minimum where each column is a sample point
#benchmarks should be a list of sample points
def split(data, benchmarks):
    n_samples = data.shape[-1]
    songs = []
    prev = 0
    for song in benchmarks:
        songs.append(data[:,prev:song])
        prev = song

    return songs

#train the l2 model (from paper)
#lam is the regularization parameter
def train_l2(X, Y, lam, samp_rate):
    n = (X.T@X).shape[0]
    k = [-np.ones(n-1),2*np.ones(n),-np.ones(n-1)]
    offset = [-1,0,1]
    M = diags(k,offset).toarray()

    M[0,0] -= 1
    M[-1, -1] -= 1
    M *= samp_rate

    W = np.linalg.inv(X.T@X + lam*M)@(X.T@Y)*samp_rate
    return W

#perform cross validation on data
#
def cross_validation(X, Y, K=1, method='ridge', alpha=1., ratio=0.5):
    scaler = StandardScaler()
    fs = 250
    assert method in ['ridge', 'l2', 'lasso', 'l1', 'elastic', 'elasticnet']

    if type(alpha) != list:
        alpha = [alpha]
    if type(ratio) != list:
        ratio = [ratio]


    if method in ['ridge', 'l2']:

        results = {f'{a}':None for a in alpha}

        for val in alpha:

            avg = []

            #leave 1 out cross val
            if False:
                pass
                num = K*-1
                #shuffle data
                ind = np.arange(len(X))
                random.shuffle(ind)
                X_shuffled = np.array(X)[ind]
                Y_shuffled = np.array(Y)[ind]
                
                for i in range(len(X)):
                    X_test = X_shuffled[i]
                    X_shuffled
                    


            #regular K-fold cross val    
            else:
                #shuffle data
                ind = np.arange(1,len(X)+1)
                random.shuffle(ind)
                X_shuffled = []
                Y_shuffled = []
                for i in ind:
                    X_shuffled.append(X[f'song{i}'])
                    Y_shuffled.append(Y[f'song{i}'])


                #split data into K roughly even subsets
                length = len(X_shuffled) // K
                for i in range(length):
                    X_test, Y_test, X_train, Y_train = [], [], [], []

                    for j in range(len(X_shuffled)):
                        if j in list(np.arange(i*K, (i+1)*K)):
                            X_test.append(X_shuffled[j])
                            Y_test.append(Y_shuffled[j].T)
                        else:
                            X_train.append(X_shuffled[j])
                            Y_train.append(Y_shuffled[j].T)
                    
                   
                    try:
                        W = 0
                        for j in range(len(X_train)):
                            if type(W) == int:
                                W = train_l2(X_train[j], Y_train[j], val, fs)
                            else:
                                W += train_l2(X_train[j], Y_train[j], val, fs)

                        W = W/len(X_train)

                        W = scaler.fit_transform(W)
                        
                        X_avg = 0
                        Y_avg = 0

                        for j in range(len(X_test)):
                            if type(X_avg) == int:
                                X_avg = X_test[j]
                                Y_avg = Y_test[j]
                            else:
                                X_avg +=  X_test[j]
                                Y_avg += Y_test[j]

                        X_avg = X_avg/len(X_test)
                        Y_avg = Y_avg/len(Y_test)

                        X_avg = scaler.fit_transform(X_avg)
                        Y_avg = scaler.fit_transform(Y_avg)

                        Y_pred = X_avg@W
                        avg.append(mean_squared_error(Y_avg, Y_pred))
                    except Exception as e:
                        print(f"\nError {e} occurred in L2 with alpha = {val}\n")
                        avg.append(np.inf)
                
                results[f'{val}'] = np.mean(avg)

    if method in ['lasso',  'l1']:
        
        results = {f'{a}':None for a in alpha}

        for val in alpha:

            avg = []

            #leave 1 out cross val
            if False:
                pass
                num = K*-1
                #shuffle data
                ind = np.arange(len(X))
                random.shuffle(ind)
                X_shuffled = np.array(X)[ind]
                Y_shuffled = np.array(Y)[ind]
                
                for i in range(len(X)):
                    X_test = X_shuffled[i]
                    X_shuffled
                    


            #regular K-fold cross val    
            else:
                #shuffle data
                ind = np.arange(1,len(X)+1)
                random.shuffle(ind)
                X_shuffled = []
                Y_shuffled = []
                for i in ind:
                    X_shuffled.append(X[f'song{i}'])
                    Y_shuffled.append(Y[f'song{i}'])


                #split data into K roughly even subsets
                length = len(X_shuffled) // K
                for i in range(length):
                    X_test, Y_test, X_train, Y_train = [], [], [], []

                    for j in range(len(X_shuffled)):
                        if j in list(np.arange(i*K, (i+1)*K)):
                            X_test.extend(X_shuffled[j])
                            Y_test.extend(Y_shuffled[j].T)
                        else:
                            X_train.extend(X_shuffled[j])
                            Y_train.extend(Y_shuffled[j].T)
                    
                    X_train = np.array(X_train)
                    X_test = np.array(X_test)
                    Y_train = np.array(Y_train)
                    Y_test = np.array(Y_test)
                    try:
                        model = Lasso(alpha=val)
                        model.fit(X_train, Y_train)
                        Y_pred = model.predict(X_test)
                        avg.append(mean_squared_error(Y_test, Y_pred))
                    except Exception as e:
                        print(f'\nError {e} occurred in L1 with alpha = {val}\n')
                        avg.append(np.inf)
                
                results[f'{val}'] = np.mean(avg)



    if method in ['elastic', 'elasticnet']:

        results = {f'{a}, {b}':None for a in alpha for b in ratio}

        for a in alpha:
            for b in ratio:
                avg = []

                #leave 1 out cross val
                if False:
                    pass
                    num = K*-1
                    #shuffle data
                    ind = np.arange(len(X))
                    random.shuffle(ind)
                    X_shuffled = np.array(X)[ind]
                    Y_shuffled = np.array(Y)[ind]
                    
                    for i in range(len(X)):
                        X_test = X_shuffled[i]
                        X_shuffled
                        


                #regular K-fold cross val    
                else:
                    #shuffle data
                    ind = np.arange(1,len(X)+1)
                    random.shuffle(ind)
                    X_shuffled = []
                    Y_shuffled = []
                    for i in ind:
                        X_shuffled.append(X[f'song{i}'])
                        Y_shuffled.append(Y[f'song{i}'])


                    #split data into K roughly even subsets
                    length = len(X_shuffled) // K
                    for i in range(length):
                        X_test, Y_test, X_train, Y_train = [], [], [], []

                        for j in range(len(X_shuffled)):
                            if j in list(np.arange(i*K, (i+1)*K)):
                                X_test.extend(X_shuffled[j])
                                Y_test.extend(Y_shuffled[j].T)
                            else:
                                X_train.extend(X_shuffled[j])
                                Y_train.extend(Y_shuffled[j].T)
                        
                        X_train = np.array(X_train)
                        X_test = np.array(X_test)
                        Y_train = np.array(Y_train)
                        Y_test = np.array(Y_test)
                        try:
                            model = ElasticNet(alpha=a, l1_ratio=b)
                            model.fit(X_train, Y_train)
                            Y_pred = model.predict(X_test)
                            avg.append(mean_squared_error(Y_test, Y_pred))
                        except Exception as e:
                            print(f'\nError {e} occurred in elasticnet with values alpha={a} and ratio={b}\n')
                    
                    results[f'{a}, {b}'] = np.mean(avg)


    return results

from sklearn.linear_model import Ridge

def train(X, Y, method='ridge', alpha=1., ratio=0.5, sample_rate=250):
    X_train, Y_train = X, Y

    fs = sample_rate
    
    if method in ['custom']:

        W_s = [train_l2(X_train[i], Y_train[i], alpha, fs) for i in range(len(X_train))]
        W = None
        for i in W_s:
            W = (W + i if not(W is None) else i)

        W = W/len(W_s)

    if method in ['ridge',  'l2']:
        model = Ridge(alpha=alpha)

        W_s = []

        for i in range(len(X_train)):
            model.fit(X_train[i], Y_train[i])
            W_s.append(model.coef_.T)

        W = None
        for i in W_s:
            W = (W + i if not(W is None) else i)

        W = W/len(W_s)

    if method in ['lasso',  'l1']:
        model = Lasso(alpha=alpha)

        W_s = []

        for i in range(len(X_train)):
            model.fit(X_train[i], Y_train[i])
            W_s.append(model.coef_.T)

        W = None
        for i in W_s:
            W = (W + i if not(W is None) else i)

        W = W/len(W_s)

    if method in ['elastic', 'elasticnet', 'elastic-net']:
        model = ElasticNet(alpha=alpha, l1_ratio=ratio)

        W_s = []

        for i in range(len(X_train)):
            model.fit(X_train[i], Y_train[i])
            W_s.append(model.coef_.T)

        W = None
        for i in W_s:
            W = (W + i if not(W is None) else i)

        W = W/len(W_s)

    return W
