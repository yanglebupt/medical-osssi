import scipy.stats as stats 
import numpy as np

def beta_pdf(α,β,loc,scale):
    def plot(x):
        return stats.beta.pdf(x,α,β,loc,scale)
    return plot

def expon_pdf(loc, scale):
    def plot(x):
        return stats.expon.pdf(x,loc,scale)
    return plot

def fit_beta(x):
    return stats.beta.fit(x)

def fit_expon(x):
    return stats.expon.fit(x)

def plot_feature(di):
    def plot(x):
        return di[x]
    return plot


def get_new_feature_by_hist(features, bins, ys, method="center", verbose=False):
    bin_centers = [(bins[i]+bins[i+1])/2 for i in range(len(ys))] 
    def get_index(x):
        if x<bin_centers[0]:
            return 0
        elif x>bin_centers[-1]:
            return len(bin_centers)-1
        else:
            for i in range(len(bin_centers)):
                bin_start = bins[i]
                bin_end = bins[i+1]
                if x<=bin_end and x>=bin_start:
                    return i
    new_features = np.empty_like(features)
    for i in range(len(features)):
        x = features[i]
        idx = get_index(x)
        if verbose:
            print(idx,x,bin_centers)
        if method=="center":
            new_features[i]=bin_centers[idx]
        elif method=="moving-average":
            pre = max(0, idx-2)
            net = min(idx+2, len(ys)-1)
            fea = 0
            total=0
            for m in range(pre, net+1):
                fea+=bin_centers[m]
                total+=1
            new_features[i]=fea/total
        else:
            pass
            
    return new_features

def check_fit(data, name, args):
    # scipy.stats.anderson
    return stats.kstest(data, name, args=args)  # 第一个值为统计量(越接近0越好)，第二个值 p 越大

def check_two_sample(sample1, sample2):
    return stats.kstest(sample1, sample2)

def prediction_probas(features, p_probas, likehood_probas, train_maxvalues, fea_range=range(16), transform_faes=None):
    n,m = features.shape
    probas = np.empty((n,2))
    for k in range(n):
        for i in range(2):
            p = p_probas[i]
            for j in fea_range:
                isNorm=j<7
                plot = likehood_probas[i,j]
                if isNorm:
                    fea_input = abs(features[k,j]) / train_maxvalues[j]
                    if transform_faes is not None:
                      transform_fae = transform_faes[i][j]
                      fea_trans = get_new_feature_by_hist(np.array([fea_input]), *transform_fae, verbose=False)
                      p_ = plot(fea_trans[0])
                    else:
                      p_ = plot(fea_input)
                else:
                    p_ = plot(features[k,j])
                p*=p_
            probas[k,i]=p
    return probas / (probas.sum(axis=1).reshape((-1,1))+1e-5)
    n,m = features.shape
    probas = np.empty((n,2))
    for k in range(n):
        for i in range(2):
            p = p_probas[i]
            for j in fea_range:
                isNorm=j<7
                plot = likehood_probas[i,j]
                if isNorm:
                    fea_input = abs(features[k,j]) / train_maxvalues[j]
                    transform_fae = transform_fae_0[j] if i==0 else transform_fae_1[j]
                    fea_trans = get_new_feature_by_hist(np.array([fea_input]), *transform_fae, verbose=False)
                    p_ = plot(fea_trans[0])
#                     p_ = plot(fea_input)
                else:
                    p_ = plot(features[k,j])
                p*=p_  
            probas[k,i]=p
    return probas / (probas.sum(axis=1).reshape((-1,1))+1e-5)