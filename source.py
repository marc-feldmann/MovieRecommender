import time
import datetime
import cmath as math
import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

### ---------------------------------------------- ###

# EXPLORATORY DATA ANALYSIS
## Import data
data_list = pd.read_table('data/ml-100k/u.data', header=0, names=['userId', 'movieId', 'rating', 'timestamp'])
data_list = data_list.drop('timestamp', axis=1)
# item_info = pd.read_table('data/ml-100k/u.item', encoding='latin-1', sep='|')
data_list.head()

## Plot mean ratings per user
temp = data_list.groupby('userId').mean()['rating'].sort_values(axis=0)
temp = temp.reset_index(level=['userId'])

height= temp['rating'].tolist()
bars = temp['userId'].astype('str').tolist()
y_pos = np.arange(len(bars))

plt.bar(y_pos, height)
plt.xlim(0, len(bars))
plt.ylim(0, 5)
plt.xlabel("User ID")
plt.ylabel("Mean Rating")
plt.title("Distribution of Mean Ratings per User")
plt.show()

## Plot rating distribution
plt.hist(data_list['rating'], bins=5)
plt.xticks([1,2,3,4,5])
plt.show()

## Plot count of missing ratings per movie
def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = 'barh')
        plt.show()
    else:
        print('No NAs found')

plot_nas(data_list.pivot(index='userId', columns='movieId', values='rating'))


# DATA PREPROCESSING
## Train/test split: 
X_train, X_test, y_train, y_test = train_test_split(data_list.iloc[:, :2], data_list.iloc[:, 2:], stratify=data_list['userId'], test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=X_train['userId'], test_size=0.2)

## Missing value imputation
### Imputation strategy: fill missing values with movies' mean ratings, subtract users' mean ratings from that (user 'bias') - to imrpove information quality in later factor matrices/during later subspace operations
# where NaN: impute with item-mean, subtract user mean
data_train_matrix = pd.concat([X_train, y_train], axis=1).pivot(index='userId', columns='movieId', values='rating').reset_index().drop('userId', axis=1)
data_train_matrix.columns = range(data_train_matrix.columns.size)
users_rating_bias = data_train_matrix.mean(axis=1)

mask = np.isnan(data_train_matrix)
masked_data_train_matrix = np.ma.masked_array(data_train_matrix, mask)
fill_value = pd.DataFrame({row: data_train_matrix.mean(axis=0) for row in data_train_matrix.index}).transpose()
data_train_matrix_imp = pd.DataFrame(masked_data_train_matrix.filled(fill_value))
data_train_matrix_imp = data_train_matrix_imp.sub(data_train_matrix.mean(axis=1), axis=0)
data_train_matrix_imp.to_csv('data/data_train_matrix_imp')

# data_train_matrix_imp = pd.read_csv('data/data_train_matrix_imp'')
data_train_matrix_imp = data_train_matrix_imp.drop('Unnamed: 0', axis=1)


# MODEL OPTIMIZATION, VALIDATION, AND SELECTION
# optimize recommender model on validation sets over grid defined by following parameters: d, k, wkernel, dist

## Define function for latent space (user-concept associations) construction during optimization
def reduction(U, s, V_t, S, V, d):

    print('Reduction started @ %s.' % datetime.datetime.now())
    t = time.time()

    s_r = s[:d]
    S_r = S[:d,:d]
    U_r = U[:,:d]
    V_r = V[:,:d]
    V_t_r = V_t[:d,:]
    US_r = np.dot(U_r, np.sqrt(S_r))
    USV_r = np.dot(np.dot(U_r, S_r), V_t_r)
    
    print('Reduction done, took %s seconds.' % round((time.time() - t), 2))
    return U_r, s_r, V_t_r, S_r, V_r, US_r, USV_r

## Define function for identifying and returning all users' neighbors in latent space during optimization
def findneighbors(X, metr, k=30):
    print('Neighborhood subroutine started @ %s.' % datetime.datetime.now())
    t = time.time()
    
    knn_model = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metr).fit(X)
    distances, neighborhoods = knn_model.kneighbors(X)
    distances = pd.DataFrame(distances).drop([0], axis=1)
    neighborhoods = pd.DataFrame(neighborhoods).drop([0], axis=1)

    print('Neighborhood construction done, took %s seconds.' % round((time.time() - t), 2))
    return distances, neighborhoods

## Define function for weighting nearest neighbors' predictions
def wfunc(i, u_neighbors_distances, wkernel):
   
    #normalize
    dist_norm = u_neighbors_distances[1]/max(u_neighbors_distances)
    
    #weight normalized distance with one of the following kernels
    if wkernel == 'triangular':
        weight = 1 - dist_norm
    elif wkernel == 'epanechnikov':
        weight = (3/4) * (1 - dist_norm**2)
    elif wkernel == 'biweight':
        weight = (15/16) * (1 - dist_norm**2)**2
    elif wkernel == 'triweight':
        weight = (35/32) * (1 - dist_norm**2)**3
    elif wkernel == 'cosine':
        weight = (math.pi/4) * math.cos((math.pi/2)*dist_norm)
    elif wkernel == 'gauss':
        weight = (1/(math.sqrt(2*(math.pi)))) * math.exp(-((dist_norm**2)/2))

    return weight


## Define functon for generating predictions (all ratings) during optimization

def predict(X, k, neighborhoods, wkernel='unweighted'):
    # X required to be of form: user_id, movie_id
    # full neighborhood matrix and reduced factor matrices need to be available before starting this function (given by functions reduce, neighborhood)

    print('Prediction subroutine started @ %s.' % datetime.datetime.now())
    t = time.time()

    preds = np.array([])
    preds_mean = np.ma.mean(np.ma.masked_equal(X, 0))

    for record in range(0, X.shape[0]):
        u = X.iloc[record, 0]
        i = X.iloc[record, 1]
        u_neighbors = neighborhoods.iloc[u-1, :k]
        u_neighbors_distances = distances.iloc[u-1, :k]

        if wkernel == 'unweighted':
            u_neighbors_ratings = np.array([])
            for i, neighbor in enumerate(u_neighbors):
                u_neighbor_rating = (np.dot(np.dot(U_r[neighbor-1, :], S_r), I_t_r[:,i-1]))
                u_neighbors_ratings = np.append(u_neighbors_ratings, u_neighbor_rating)

            pred = sum(u_neighbors_ratings)/np.count_nonzero(u_neighbors_ratings)
        
        elif wkernel == 'simple':
            u_neighbors_weights = np.array([])
            u_neighbors_ratings = np.array([])
            for i, neighbor in enumerate(u_neighbors):
                u_neighbor_rating = (np.dot(np.dot(U_r[neighbor-1, :], S_r), I_t_r[:,i-1]))
                u_neighbor_weight = 1 - (u_neighbors_distances.iloc[i]/(sum(u_neighbors_distances.iloc[:])+0.00001))
                u_neighbors_ratings = np.append(u_neighbors_ratings, u_neighbor_rating)
                u_neighbors_weights = np.append(u_neighbors_weights, u_neighbor_weight)
            
            u_neighbors_weights_nrm = u_neighbors_weights/(sum(u_neighbors_weights)+0.00001) # normalize all batch neighbors weights to 1
            pred = sum(u_neighbors_ratings * u_neighbors_weights_nrm)
            
        else:
            u_neighbors_weights = np.array([])
            u_neighbors_ratings = np.array([])
            for i, neighbor in enumerate(u_neighbors):
                u_neighbor_rating = (np.dot(np.dot(U_r[neighbor-1, :], S_r), I_t_r[:,i-1]))
                u_neighbor_weight = wfunc(i, u_neighbors_distances, wkernel)
                u_neighbors_ratings = np.append(u_neighbors_ratings, u_neighbor_rating)
                u_neighbors_weights = np.append(u_neighbors_weights, u_neighbor_weight)

            u_neighbors_weights_nrm = u_neighbors_weights/(sum(u_neighbors_weights)+0.00001) # normalize all batch neighbors weights to 1
            pred = sum(u_neighbors_ratings * u_neighbors_weights_nrm)   
            
        pred += users_rating_bias[u-1] #re-add mean of active user (= his overall rating mean)
        pred = pred.real
        if np.isnan(pred) == True:
            pred =  preds_mean # opt potenzial: change to: item mean + user bias
        preds = np.append(preds, pred)

    print('Prediction done, took %s seconds, used %s kernel' % (round((time.time() - t), 2), wkernel))
    return preds

## ----------- Grid search 1: optimizing k, d (dist=euclidean, kNN-weighting: unweighted) ----------- ##
dt = datetime.datetime.now()
print('Optimization of latent space neigbor recommender started at %s' % dt)
start_time = time.time()

print('Gridsearch over k and d started @ %s.' % datetime.datetime.now())
t = time.time()

## Decompose data into factor matrices and singular value matrix
U, s, I_t = svd(data_train_matrix_imp, full_matrices=False)
S = np.diag(s)
I = np.transpose(I_t)

## Plot percent variance explained by singular values
var_explained = np.round(s**2/np.sum(s**2), decimals=3)
sns.barplot(x=list(range(1,len(var_explained)+1)), y=var_explained, color="limegreen")
plt.xlabel('Singular Values', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.xlim(0, 50)
plt.ylim(0, 0.15)
plt.show()

#Parameter Loop and Plotting
mae_values = np.array([])
mae_values_min = float('inf')
d_values = np.array([])
k_values = np.array([])
lgridname = []
lrecerror = []
for d in range(100, 0, -5):
    print('d = %s iteration started @ %s.' % (d, datetime.datetime.now()))
    t = time.time()
    
    U_r, s_r, I_t_r, S_r, I_r, US_r, USI_r = reduction(U, s, I_t, S, I, d)
    distances, neighborhoods = findneighbors(X=US_r, k=data_list['userId'].nunique()-1, metr='euclidean')
    
    for k in range(2, 52, 2):
        print('k = %s iteration started @ %s.' % (k, datetime.datetime.now()))
        y_val_preds = predict(X_val, k, neighborhoods, wkernel='unweighted')
        
        mae_value = mean_absolute_error(y_val_preds, y_val)
        mae_values = np.append(mae_values, mae_value)
        d_values = np.append(d_values, d)
        k_values = np.append(k_values, k)
        
        if min(mae_values) < mae_values_min:
            mae_values_min = min(mae_values)
            mae_values_opt_d = d
            mae_values_opt_k = k
        
        print('%s neighbors iteration complete.' % k)
        lrecerror.append(norm((data_train_matrix_imp - USI_r), ord='fro')/norm((data_train_matrix_imp), ord='fro'))
    
    print('%s dimensions iteration complete.' % d)

#store opt param from this grid search for next grid search
mae_values_min_eucl = min(mae_values)
    
#Plotting:
x = d_values
y = k_values
z = mae_values

fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.set_xlabel('subspace dimensions d', fontsize=10)
ax.set_ylabel('nearest neighbors k', fontsize=10)
ax.set_zlabel('MAE', fontsize=10)
ax.set_xlim3d([0,max(d_values)])
ax.set_ylim3d([0,max(k_values)])
#ax.set_zlim3d([min(mae_values)-0.05,max(mae_values)+0.05])

ax = ax.plot_trisurf(x, y, z, linewidth=0.5, cmap=cm.coolwarm, antialiased=True)

filename = 'kNN_rec_gridsearch_kd_' + datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
plt.savefig('results/' + filename + '.png')
plt.close()

print('Gridsearch over k and d done, took %s seconds.' % round((time.time() - t), 2))
print('Gridsearch Protocol:')
print('  min MAE = %.4f' %mae_values_min)
print('    @optimal k = %d' % mae_values_opt_k)
print('    @optimal d = %d' % mae_values_opt_d)

#Preparing protocol info
ld = x.tolist()
lk = y.tolist()
lmae = z.tolist()
lgridname = ['Grid 1 (k, d)'] * len(lmae)
lmetr = ['euclidean'] * len(lgridname)
lweight = ['none'] * len(lgridname)
lkmax = [max(k_values)]  * len(lgridname)
lkmin = [min(k_values)] * len(lgridname)
lkstep = [max(k_values) / len(set(lk))] * len(lgridname)
ldmax = [max(d_values)] * len(lgridname)
ldmin = [min(d_values)] * len(lgridname)
ldstep = [max(d_values) / len(set(ld))] * len(lgridname)
lgriddur = [(round((time.time() - t), 2))] * len(lgridname)


## ----------- Grid search 2: optimizing distance metric (k=opt, d=opt, kNN-weighting=unweighted) ----------- ##
metrics = ['cosine', 'correlation', 'cityblock', 'mahalanobis', 'canberra']

print('Grid search over distance metrics started @ %s.' % datetime.datetime.now())
t = time.time()

#set minimum error from last grid search as benchmark for this grid search
mae_values_min = mae_values_min_eucl
mae_values_opt_m = 'euclidean'

#initialize empty 'value buckets' to be filled during iterations
metric_values = np.array([])
mae_values = np.array([])

U_r, s_r, I_t_r, S_r, I_r, US_r, USI_r = reduction(U, s, I_t, S, I, mae_values_opt_d)
for i in metrics:
    distances, neighborhoods = findneighbors(X=US_r, k=mae_values_opt_k, metr='euclidean')
    y_val_preds = predict(X=X_val, k=mae_values_opt_k, neighborhoods=neighborhoods, wkernel='unweighted')
        
    mae_value = mean_absolute_error(y_val_preds, y_val)
    mae_values = np.append(mae_values, mae_value)
    metric_values = np.append(metric_values, i)
    
    if min(mae_values) < mae_values_min:
        mae_values_min = min(mae_values)
        mae_values_opt_m = i
    
    print('%s metric iteration complete.' % i)

#store opt param from this grid search for next grid search
mae_values_min_unw = min(mae_values)

print('Gridsearch over distance metrics done, took %s seconds.' % round((time.time() - t), 2))
print('Gridsearch Protocol:')
print('  min MAE = %.4f' %mae_values_min)
print('    @optimal m = %s' % mae_values_opt_m)

#Preparing protocol info
ld.extend([mae_values_opt_d] * len(metrics))
lk.extend([mae_values_opt_k] * len(metrics))
lmae.extend(mae_values.tolist())
lgridname.extend(['Grid 2 (dist metric)'] * len(metrics))
lmetr.extend(metric_values.tolist())
lweight.extend(['none'] * len(metrics))
lkmax.extend(['n.a.'] * len(metrics))
lkmin.extend(['n.a.'] * len(metrics))
lkstep.extend(['n.a.'] * len(metrics))
ldmax.extend(['n.a.'] * len(metrics))
ldmin.extend(['n.a.'] * len(metrics))
ldstep.extend(['n.a.'] * len(metrics))
lgriddur.extend([(round((time.time() - t), 2))] * len(metrics))
lrecerror.extend([lrecerror[ld.index(mae_values_opt_d)]] * len(metrics))

## ----------- Grid search 3: optimizing kNN-weighting kernel (k=opt, d=opt, metric=opt) ----------- ##
kernels = ['simple', 'epanechnikov', 'biweight', 'triweight', 'cosine', 'gauss']

print('Gridsearch over kNN-weighting kernels started @ %s.' % datetime.datetime.now())
t = time.time()

#set minimum error from last grid search as benchmark for this grid search
mae_values_min = mae_values_min_unw
mae_values_opt_ws = 'unweighted'

#initialize empty 'value buckets' to be filled during iterations
kernel_values = []
mae_values = np.array([])

U_r, s_r, I_t_r, S_r, I_r, US_r, USI_r = reduction(U, s, I_t, S, I, mae_values_opt_d)
distances, neighborhoods = findneighbors(X=US_r, k=mae_values_opt_k, metr=mae_values_opt_m)
for i in kernels:
    y_val_preds = predict(X=X_val, k=mae_values_opt_k, neighborhoods=neighborhoods, wkernel=i)
        
    mae_value = mean_absolute_error(y_val_preds, y_val)
    mae_values = np.append(mae_values, mae_value)
    kernel_values = np.append(kernel_values, i)
    
    if min(mae_values) < mae_values_min:
        mae_values_min = min(mae_values)
        mae_values_opt_ws = i        
    
    print('%s kernel iteration complete.' % i)

#store opt param from this grid search for next grid search
mae_values_min_rat02 = min(mae_values)

print('Gridsearch over kNN-weighting kernels done, took %s seconds.' % round((time.time() - t), 2))
print('Gridsearch Protocol:')
print('  min MAE = %.4f' %mae_values_min)
print('    @optimal kernel = %s' % mae_values_opt_ws)

#Preparing protocol info
ld.extend([mae_values_opt_d] * len(kernels))
lk.extend([mae_values_opt_k] * len(kernels))
lmae.extend(mae_values.tolist())
lgridname.extend(['Grid 3 (kNN weighting kernel)'] * len(metrics))
lmetr.extend([mae_values_opt_m] * len(kernels))
lweight.extend(kernel_values.tolist())
lkmax.extend(['n.a.'] * len(kernels))
lkmin.extend(['n.a.'] * len(kernels))
lkstep.extend(['n.a.'] * len(kernels))
ldmax.extend(['n.a.'] * len(kernels))
ldmin.extend(['n.a.'] * len(kernels))
ldstep.extend(['n.a.'] * len(kernels))
lgriddur.extend([(round((time.time() - t), 2))] * len(kernels))
lrecerror.extend([lrecerror[ld.index(mae_values_opt_d)]] * len(kernels))


## ----------- Create protocol ----------- ##
ltotdur = [round((time.time() - start_time), 2)] * len(lgridname)

# create DataFrame titled and store to specified location
L = pd.DataFrame([ltotdur, lrecerror, lgridname, lmae, lk, lkmax, lkmin, lkstep, ld, ldmax, ldmin, ldstep, lmetr, lweight])
L = L.T
L.columns = ['total duration', 'rel_recon_error', 'grid', 'MAE', 'k', 'k_max', 'k_min', 'k_step', 'd', 'd_max', 'd_min', 'd_step', 'metric', 'weighting_scheme']
filename = 'kNN_rec_optimizationlog_' + datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S') + '.csv'
L.to_csv(path_or_buf=('results/' + filename), sep=';', index=False, decimal=',')

print('Optimization of latent space neighbor recommender took %s seconds.' % round((time.time() - start_time), 2))

