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
from plain_methods.nb import *

fit_models_nb = [
  [fit_beta,beta_pdf],
  [fit_expon,expon_pdf],
  [fit_expon,expon_pdf],
  [fit_beta,beta_pdf],
  [fit_expon,expon_pdf],
  [fit_beta,beta_pdf],
  [fit_beta,beta_pdf]
]

best_bins = [
    [50,15],
    [100,75],
    [65,25],
    [10,10],
    [100,20],
    [200,185],
    [20,15]
]

fit_models_nb_bins = [
    [fit_beta,beta_pdf],
    [fit_expon,expon_pdf],
    [fit_expon,expon_pdf],
    [fit_beta,beta_pdf],
    [fit_expon,expon_pdf],
    [fit_beta,beta_pdf],
    [fit_beta,beta_pdf]
]

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--val_filepath', type=str, default='./data/training1109.xlsx', help='输入验证数据集路径')
    parser.add_argument('--save_filename', type=str, default='results', help='结果输出文件名')
    args = parser.parse_args()
    return args

def nb_test(features, labels):
    rootdirs = ["./plain_methods/pre-trained/nb", "./plain_methods/pre-trained/nb-bins"]
    rocs = []
    for rootdir in rootdirs:
      model_and_traces_0 = []
      model_and_traces_1 = []
      p_probas = np.loadtxt(os.path.join(rootdir, "p_probas.txt"))
      maxvalues = np.loadtxt(os.path.join(rootdir, "maxvalues.txt"))
      traces_0 = np.load(os.path.join(rootdir, "traces_0.npy"), allow_pickle=True)
      traces_1 = np.load(os.path.join(rootdir, "traces_1.npy"), allow_pickle=True)
      if "nb-bins" in rootdir:
        transform_fae_0 = np.load(os.path.join(rootdir, "transform_fae_0.npy"), allow_pickle=True)
        transform_fae_1 = np.load(os.path.join(rootdir, "transform_fae_1.npy"), allow_pickle=True)
        transform_faes = [transform_fae_0,transform_fae_1]
        method="nb-bins"
      else:
        method="nb"
        transform_faes = None

      for i, (_, pdf_m) in enumerate(fit_models_nb):
        plot_0 = pdf_m(*traces_0[i])
        plot_1 = pdf_m(*traces_1[i])
        model_and_traces_0.append(plot_0)
        model_and_traces_1.append(plot_1)
      for i in range(7,16):
        model_and_traces_0.append(plot_feature(traces_0[i]))
        model_and_traces_1.append(plot_feature(traces_1[i]))
      
      likehood_probas = np.array([
        model_and_traces_0,
        model_and_traces_1,
      ],dtype="object")

      scores = prediction_probas(features, p_probas, likehood_probas, train_maxvalues=maxvalues, transform_faes=transform_faes)
      save_scores(TMP_PATH + "/" + args.save_filename, "nb", scores, method, labels=labels)
      score_roc_auc = roc_auc_score(labels, scores[:,1])
      print(f"method: {rootdir.split('/')[-1]}, roc: {score_roc_auc}")
      rocs.append(score_roc_auc)

    results = dict(model_names=[f.split("/")[-1] for f in rootdirs], roc=rocs)
    save_xslx(OUTPUT_PATH, args.save_filename, results)

if __name__ == "__main__":
  args = parse_args_and_config()
  features, labels = read_fea_label(args.val_filepath, usedHeaders, dtype)
  fea_1, fea_2 = get_fea16(features, var_pre_post, var_one_2)
  selected_features = np.concatenate([fea_1,fea_2],axis=1)
  config = yaml.safe_load("./plain_methods/config.yaml")

  make_dir([OUTPUT_PATH, TMP_PATH, TMP_PATH + "/" + args.save_filename])
  nb_test(selected_features, labels)

  