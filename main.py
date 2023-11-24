import argparse
from tools import *
from constants import *
from torch.utils.data import DataLoader
from models import MyDataset
from pathmap import *
import torch
from test import * 
from train import *
import yaml
import os
from sklearn.metrics import roc_auc_score

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--val_filepath', type=str, default='./data/training1109.xlsx', help='输入验证数据集路径')
    parser.add_argument('--save_filename', type=str, default='results', help='结果输出文件名')
    parser.add_argument('--type', type=str, default='all-features', help='all-features/fea16/pre-surg')
    args = parser.parse_args()
    return args


def find_index_epoch_by_filename(filename,epoches):
    for i in range(len(epoches)-1, -1, -1):
      if str(epoches[i]) in filename:
        return i
      
def filter_model_filename(filenames,epoches):
    filtered_files = [file for file in filenames if file.endswith(MODEL_EXTENSION)]
    # 合并相同的 epoch，并排序
    res_files = []
    for epoch in epoches:
      epoch_files = [f for f in filtered_files if "_" + str(epoch) + MODEL_EXTENSION in f]
      if(len(epoch_files)>0):
        res_files.append(epoch_files)
    return res_files

if __name__ == "__main__":
  args = parse_args_and_config()
  all_features, all_labels = read_fea_label(args.val_filepath, usedHeaders, dtype)

  modelType = args.type
  pre_trained_root = os.path.join(PRETRAINED_PATH, modelType) 
  if modelType=="all-features":
    print("使用全部特征")
    selected_features = all_features
  elif modelType=="fea16":
    print("使用筛选的16个特征")
    fea_1, fea_2 = get_fea16(all_features, var_pre_post, var_one_2)
    selected_features = np.concatenate([fea_1,fea_2],axis=1)
  elif modelType=="pre-surg":
    print("使用术前特征")
    used_columns = [usedHeaders.index(n) for n in pre_surg_headers]
    selected_features = all_features[:, used_columns]
  else:
     pass
  maxvalues_all = np.max(np.abs(selected_features),axis=0)
  maxvalues_all_default = np.loadtxt(os.path.join(pre_trained_root, "maxvalues.txt"))
  
  maxvalues_all_merge = np.max(
    np.concatenate([maxvalues_all.reshape((1,-1)),maxvalues_all_default.reshape((1,-1))],axis=0),
    axis=0
  )

  fea_dataset = MyDataset(selected_features, all_labels, maxvalues_all_default)
  fea_dataloader = DataLoader(fea_dataset, batch_size=10, shuffle=False)

  torch.set_default_tensor_type(torch.FloatTensor)

  # 开始预测
  epoches_list = EPOCHES_ALL
  model_list_path = [i for i in os.listdir(pre_trained_root) if os.path.isdir(
     os.path.join(pre_trained_root,i)
  )]

  all_epoch_results = np.empty((len(model_list_path), len(epoches_list)), dtype=object)
  all_epoch_results.fill("")

  pd_names = []
  make_dir([OUTPUT_PATH, TMP_PATH, TMP_PATH + "/" + args.save_filename])
  for i, model_path in enumerate(model_list_path):  # 遍历模型
    folder = os.path.join(pre_trained_root, model_path)
    model_epoch_filenames = filter_model_filename(os.listdir(folder), epoches_list)
    pd_names.append(model_path)
    for j, model_filename_list in enumerate(model_epoch_filenames): # 遍历epoch
      score_roc_auc_list = []
      for model_filename in model_filename_list:
        model = create_model_by_name(model_path)
        checkpoint = torch.load(os.path.join(folder, model_filename))
        model.load_state_dict(checkpoint["model"])
        model = model.eval()
        conv1d = "cnn1d" in model_path
        error, scores, labels = test_proba(device, fea_dataloader, model, loss_fn, conv1d=conv1d)
        save_scores(TMP_PATH + "/" + args.save_filename, model_path, scores, model_filename)
        print(f"pre-trained_model_path: {model_path}, model_eopch: {model_filename}, error: {error}")
        scores_one = scores[:,1]
        score_roc_auc = roc_auc_score(labels, scores_one)
        score_roc_auc_list.append(score_roc_auc)
        c_idx = find_index_epoch_by_filename(model_filename,epoches_list)
        
      all_epoch_results[i,c_idx] = np.array(score_roc_auc_list).mean()

  results = dict(model_names=pd_names)
  for i in range(len(epoches_list)):
    results[f"epoch_{epoches_list[i]}"] = all_epoch_results[:,i]

  save_xslx(OUTPUT_PATH, args.save_filename, results)

