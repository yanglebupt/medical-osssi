import argparse
from tools import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score 
from sklearn.svm import SVC
import joblib
from sklearn.linear_model import LogisticRegression
from plain_methods.rvc import RVC
import pickle
import yaml
from pathmap import *
from constants import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--val_filepath', type=str, default='./data/training1109.xlsx', help='输入验证数据集路径')
    parser.add_argument('--save_filename', type=str, default='results', help='结果输出文件名')
    args = parser.parse_args()
    return args

def svc_rvc_lr_test(features, labels):
    print(features[:10,:], labels[:10])
    rootdir = "./plain_methods/pre-trained/svc-rvc-lr"
    scaler = pickle.load(open(os.path.join(rootdir, "scaler.pkl"), 'rb'))
    features_s = scaler.transform(features)
    rocs = []
    model_filenames = [f for f in os.listdir(rootdir) if f.endswith(".model")]
    for model_filename in model_filenames:
      model = joblib.load(os.path.join(rootdir, model_filename))
      scores = model.predict_proba(features_s)[:,1]
      score_roc_auc = roc_auc_score(labels, scores)
      print(f"model_name: {model_filename}, roc: {score_roc_auc}")
      rocs.append(score_roc_auc)

    results = dict(model_names=[f.split(".model")[0] for f in model_filenames],roc=rocs)
    save_xslx(OUTPUT_PATH, args.save_filename, results)

if __name__ == "__main__":
  args = parse_args_and_config()
  features, labels = read_fea_label(args.val_filepath, usedHeaders, dtype)
  fea_1, fea_2 = get_fea16(features, var_pre_post, var_one_1)
  selected_features = np.concatenate([fea_1,fea_2],axis=1)
  config = yaml.safe_load("./plain_methods/config.yaml")

  svc_rvc_lr_test(selected_features, labels)

  