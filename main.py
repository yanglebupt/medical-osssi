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
  maxvalues_all_default = np.loadtxt(os.path.join(PRETRAINED_PATH, "maxvalues.txt"))
  
  maxvalues_all_merge = np.max(
    np.concatenate([maxvalues_all.reshape((1,-1)),maxvalues_all_default.reshape((1,-1))],axis=0),
    axis=0
  )

  fea_dataset = MyDataset(all_features, all_labels, maxvalues_all_default)
  fea_dataloader = DataLoader(fea_dataset, batch_size=10, shuffle=True)

  torch.set_default_tensor_type(torch.FloatTensor)

  # 开始预测
  epoches_list = EPOCHES_ALL
  model_list_path = [i for i in os.listdir(PRETRAINED_PATH) if os.path.isdir(
     os.path.join(PRETRAINED_PATH,i)
  )]

  all_epoch_results = np.empty((len(model_list_path), len(epoches_list)), dtype=object)
  all_epoch_results.fill("")

  pd_names = []
  for i, model_path in enumerate(model_list_path):  # 遍历模型
    folder = os.path.join(PRETRAINED_PATH, model_path)
    model_epoch_filenames = filter_model_filename(os.listdir(folder), epoches_list)
    pd_names.append(model_path)
    print(model_epoch_filenames)
    for j, model_filename_list in enumerate(model_epoch_filenames): # 遍历epoch
      score_roc_auc_list = []
      for model_filename in model_filename_list:
        model = create_model_by_name(model_path)
        checkpoint = torch.load(os.path.join(folder, model_filename))
        model.load_state_dict(checkpoint["model"])
        model = model.eval()
        conv1d = "cnn1d" in model_path
        error, scores, labels = test_proba(device, fea_dataloader, model, loss_fn, conv1d=conv1d)
        print(f"pre-trained_model_path: {model_path}, model_eopch: {model_filename}, error: {error}")
        scores_one = scores[:,1]
        score_roc_auc = roc_auc_score(labels, scores_one)
        score_roc_auc_list.append(score_roc_auc)
        c_idx = find_index_epoch_by_filename(model_filename,epoches_list)
        
      all_epoch_results[i,c_idx] = np.array(score_roc_auc_list).mean()

  results = dict(model_names=pd_names)
  for i in range(len(epoches_list)):
    results[f"epoch_{epoches_list[i]}"] = all_epoch_results[:,i]

  sheet = pd.DataFrame(results)

  result_filename = f"{OUTPUT_PATH}/{args.save_filename}.xlsx"
  exists = os.path.exists(result_filename)
  mode = "a" if exists else "w"
  kwargs={
    "mode":mode,
    "if_sheet_exists":"replace"
  }if exists else {
    "mode":mode
  }
  with pd.ExcelWriter(result_filename, **kwargs) as xlsx:
    print(f"save sheet_name")
    sheet.to_excel(xlsx, index=False)

