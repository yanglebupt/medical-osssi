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
    parser.add_argument('--val_filepath', type=str, default='./data/training0925.xlsx', help='输入验证数据集路径')
    args = parser.parse_args()
    return args


def find_index_epoch_by_filename(filename,epoches):
   for i in range(len(epoches)):
      if str(epoches[i]) in filename:
         return i
      
def filter_model_filename(filenames,epoches):
    filtered_files = [file for file in filenames if file.endswith(MODEL_EXTENSION)]
    # 合并相同的 epoch，并排序
    res_files = []
    for epoch in epoches:
      epoch_files = [f for f in filtered_files if str(epoch) in f]
      if(len(epoch_files)>0):
        res_files.append(epoch_files)
    return res_files

if __name__ == "__main__":
  args = parse_args_and_config()
  values, headers, nullIndexs = readExcel(args.val_filepath)
  rmRowIndexs = list(set(list(nullIndexs[0])))
  print(f"删除了{len(rmRowIndexs)}行空值数据")
  
  all_features = getFeatures(values, headers, usedHeaders=usedHeaders, dtype=dtype)
  all_features = np.delete(all_features,rmRowIndexs,0)
  print("全部特征", all_features.shape, all_features.dtype)
  print(all_features)

  all_labels = getLabels(values, headers, labelHeader="ssi.bin", dtype="int")
  all_labels = np.delete(all_labels,rmRowIndexs,0)
  print("全部标签", all_labels.shape, all_labels.dtype)
  print(all_labels)

  print(len(headers_1)+len(headers_2)+len(headers_3)==len(usedHeaders))

  # 全部特征
  print("使用全部特征")
  maxvalues_all = np.max(np.abs(all_features),axis=0)
  maxvalues_all_default = np.loadtxt(os.path.join(FEA_ALL , "max_value.txt"))
  
  maxvalues_all_merge = np.max(
    np.concatenate([maxvalues_all.reshape((1,-1)),maxvalues_all_default.reshape((1,-1))],axis=0),
    axis=0
  )

  fea_all_dataset = MyDataset(all_features, all_labels, maxvalues_all_merge)
  fae_all_dataloader = DataLoader(fea_all_dataset, batch_size=10, shuffle=True)

  # 16个特征
  print("使用16个特征")
  fea_1, fea_2 = get_fea16(all_features)
  all_selected_features = np.concatenate([fea_1,fea_2],axis=1)
  maxvalues_selected = np.max(np.abs(all_selected_features),axis=0)
  maxvalues_selected_default = np.loadtxt(FEA_16 + "max_value.txt")

  maxvalues_selected_merge = np.max(
    np.concatenate([maxvalues_selected.reshape((1,-1)),maxvalues_selected_default.reshape((1,-1))],axis=0),
    axis=0
  )

  fea_16_dataset = MyDataset(all_selected_features, all_labels, maxvalues_selected_merge)
  fae_16_dataloader = DataLoader(fea_16_dataset, batch_size=10, shuffle=True)

  torch.set_default_tensor_type(torch.FloatTensor)

  for i in range(2):
      # 开始预测
      fea_path = FEA_ALL if i==0 else FEA_16
      fea_dataloader = fae_all_dataloader if i==0 else fae_16_dataloader
      out_path = OUTPUT_ALL if i==0 else OUTPUT_16
      create_model_by_name = create_fea45_model_by_name if i==0 else create_fea16_model_by_name
      epoches_list = EPOCHES_ALL
      make_dir(out_path)
      with open(os.path.join(fea_path,"config.yaml"),"r") as f:
          config = yaml.safe_load(f)
      config = dict2namespace(config)
      for rate_path in config.rates:      
        pd_names = []
        model_config = config.models.__dict__
        if rate_path not in model_config and "common" in model_config:
          rate_model_config = model_config["common"]
        elif rate_path in model_config and "common" in model_config:
          rate_model_config =  model_config[rate_path] + model_config["common"]
        elif rate_path in model_config and "common" not in model_config:
          rate_model_config =  model_config[rate_path]
        else:
           continue
        all_epoch_results = np.empty((len(rate_model_config), len(epoches_list)), dtype=object)
        all_epoch_results.fill("")

        for i, model_config in enumerate(rate_model_config):  # 遍历模型
          model_config = dict2namespace(model_config)
          folder = os.path.join(fea_path, rate_path, model_config.path)
          model_epoch_filenames = filter_model_filename(os.listdir(folder), epoches_list)
          pd_names.append(model_config.name)
          for j, model_filename_list in enumerate(model_epoch_filenames): # 遍历epoch
            score_roc_auc_list = []
            for model_filename in model_filename_list:
              model = create_model_by_name(model_config.name)
              checkpoint = torch.load(os.path.join(folder, model_filename))
              model.load_state_dict(checkpoint["model"])
              model = model.eval()
              conv1d = model_config.conv1d if "conv1d" in model_config else False
              error, scores, labels = test_proba(device, fea_dataloader, model, loss_fn, conv1d=conv1d)
              print(f"train_test_rate: {rate_path}, pre-trained_model_path: {model_config.path}, model_eopch: {model_filename}, error: {error}")
              scores_one = scores[:,1]
              score_roc_auc = roc_auc_score(labels, scores_one)
              score_roc_auc_list.append(score_roc_auc)
              c_idx = find_index_epoch_by_filename(model_filename,epoches_list)
              
            all_epoch_results[i,c_idx] = np.array(score_roc_auc_list).mean()
            # str_list = [str(s) for s in score_roc_auc_list]
            # all_epoch_results[i,j] = ",".join(str_list)

        rate_results = dict(model_names=pd_names)
        for i in range(len(epoches_list)):
          rate_results[f"epoch_{EPOCHES_ALL[i]}"] = all_epoch_results[:,i]

        rate_sheet = pd.DataFrame(rate_results)

        result_filename = f"{out_path}/results.xlsx"
        if os.path.exists(result_filename):
          with pd.ExcelWriter(result_filename, mode="a", if_sheet_exists="replace") as xlsx:
            print(f"save sheet_name: {rate_path}")
            rate_sheet.to_excel(xlsx, sheet_name=rate_path, index=False)
        else:
          with pd.ExcelWriter(result_filename, mode="w") as xlsx:
            print(f"save sheet_name: {rate_path}")
            rate_sheet.to_excel(xlsx, sheet_name=rate_path, index=False)

