# medical-osssi

设计的网络结构如下来预测 osssi 结局变量

### 请按照下述过程将模型用于推理

- 安装 python，并使用下面的命令安装依赖

```bash
pip install -r requirement.lock.txt
```

- 执行下面的命令进行模型推理，`val_filepath` 参数指定验证数据集路径

```bash
python main.py --val_filepath="./data/validation1109.xlsx" --save_filename="ann-cnn1d"

python main.py --val_filepath="./data/validation1109.xlsx" --type="fea16" --save_filename="cnn1d-fea16"

python main.py --val_filepath="./data/validation1109.xlsx" --type="pre-surg" --save_filename="cnn1d-pre-surg"

python svc-rvc-lr-main.py --val_filepath="./data/validation1109.xlsx" --save_filename="svc-rvc-lr"

python nb-main.py --val_filepath="./data/validation1109.xlsx" --save_filename="nb"
```

最终的结果会输出到 `output` 路径下

### 目前的结果

|  method   | Train ROC | Val ROC
|  ----  | ---- | ----  |
| LG  | 0.744961 | 0.732713 |
| SVC_Linear | 0.744613 | 0.731084 |
| SVC_RBF | 0.739970 | 0.724891 |
| RVC_Linear | 0.741727 | 0.736775 |
| RVC_RBF | 0.757279 | 0.736900 |
| GLM | 0.740654 | 0.733039 |
| Naive Bayes | 0.721975 | 0.711904 |
| Naive Bayes (bins) | 0.705781 | 0.713157 |
| ANN | 0.803266 | 0.736100 |
| CNN1d-Fit | 0.793942 | 0.759464 |
| CNN1d-Half (Avg10) | 0.796082 $\pm$ 0.007403 | 0.762683 $\pm$ 0.007819 |
| CNN1d-Half (Train Max) | 0.810856 | 0.752119 |
| CNN1d-Half (Val Max) | 0.793128 | 0.777190 |


### 不同特征下 CNN1d-Half 的结果 

|  Features   | Train ROC | Val ROC
|  ----  | ---- | ----  |
| fea16 (Avg10) | 0.761335 $\pm$ 0.017303 | 0.725576 $\pm$ 0.004856 |
| fea16 (Train Max) | 0.798262 | 0.728025 |
| fea16 (Val Max) | 0.767363 | 0.737978 |
| pre-surg (Avg10) | 0.755674 $\pm$ 0.009100 | 0.670600 $\pm$ 0.012963 |
| pre-surg (Train Max) | 0.767516 | 0.670210 |
| pre-surg (Val Max) | 0.760289 | 0.697663 |
| all (Avg10) | 0.796082 $\pm$ 0.007403 | 0.762683 $\pm$ 0.007819 |
| all (Train Max) | 0.810856 | 0.752119 |
| all (Val Max) | 0.793128 | 0.777190 |