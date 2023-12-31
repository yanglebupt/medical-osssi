import numpy as np
import pandas as pd
from constants import var_pre_post,var_one,usedHeaders
import argparse
import os

def make_dir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        return
    os.mkdir(dir)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def readExcel(filename, sheet_name="Sheet1", usecols=None):
    df=pd.read_excel(io=filename,sheet_name=sheet_name, usecols=usecols)
    nullIndexs = np.where((df.isnull()).values)
    return df.values, list(df.columns), nullIndexs


def getFeatures(values, headers, usedHeaders=None, dtype="float32"):
    if usedHeaders is None:
        return values
    usedHeaderIndexs = [headers.index(h) for h in usedHeaders]
    data = np.array(values[:,usedHeaderIndexs], dtype=dtype)
    return data

def getLabels(values, headers, labelHeader=None, dtype="int"):
    if labelHeader is None:
         raise Exception(f"必须指定labelHeader参数")
    labelHeaderIdx = headers.index(labelHeader)
    labels = np.array(values[:,labelHeaderIdx], dtype=dtype)
    return labels 


"""
ctype:
  - both 取术前和术后
  - pre 只取术前
  - post 只取术后
  - rate 术后/术前
  - change 术后-术前
  - ratio (术后-术前)/术前 * 100
islog 是否取对数
"""
def calcPrePost(features, ctype="post", islog=False):
    pre = features[:,0]
    post = features[:,1]
    res=None
    if ctype=="both":
        res = np.empty((features.shape[0],2))
        res[:,0]=pre
        res[:,1]=post
    elif ctype=="pre":
        res = pre
        res = res.reshape((-1,1))
    elif ctype=="post":
        res = post
        res = res.reshape((-1,1))
    elif ctype=="rate":
        res = post/pre
        res = res.reshape((-1,1))
    elif ctype=="change":
        res = post-pre
        res = res.reshape((-1,1))
    elif ctype=="ratio":
        res = 100* (post-pre)/pre
        res = res.reshape((-1,1))
    else:
        res = features
    
    return np.log(res) if islog else res


def get_fea16(all_features):
    fea_1 = None
    fea_1_names = []
    select_1_columns = []
    for pair in var_pre_post:
        columns = [usedHeaders.index(pair["pre"]), usedHeaders.index(pair["post"])]
        select_1_columns += [usedHeaders.index(pair["pre"]), usedHeaders.index(pair["post"])]
        fea = calcPrePost(
            all_features[:,columns], 
            ctype=pair["ctype"],
            islog=pair["islog"]
        )
        fea_1 = fea if fea_1 is None else np.concatenate([fea_1,fea],axis=1)
        if fea.shape[1]>1:
            fea_1_names.append(pair["rename"]+"-pre")
            fea_1_names.append(pair["rename"]+"-post")
        else:
            fea_1_names.append(pair["rename"])

    fea_2 = None
    fea_2_names = []
    select_2_columns = []
    for key_func in var_one:
        if not "func" in key_func:
            column = usedHeaders.index(key_func["key"])
            fea = all_features[:,column].reshape((-1,1))
        else:
            key = key_func["key"]
            func = key_func["func"]
            column = usedHeaders.index(key)
            fea = all_features[:,column]
            fea = func(fea).reshape((-1,1))
        select_2_columns += [column]    
        fea_2 = fea if fea_2 is None else np.concatenate([fea_2,fea],axis=1)
        if fea.shape[1]>1:
            fea_2_names.append(key_func["rename"]+"-pre")
            fea_2_names.append(key_func["rename"]+"-post")
        else:
            fea_2_names.append(key_func["rename"])

    return fea_1, fea_2