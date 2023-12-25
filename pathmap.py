EPOCHES_ALL = [75,150,200,250,300,400,500,700,800,1000,1200,1500,2000]
EPOCHES_ALL_2 = [10,20,30,40,50,60,70,100,150,200,250,300,350,400]
EPOCHES_ALL_3 = [75,150,200,250,300,400,500,700,800,1000,1200]

def get_epoch_list(epoch_nums):
  if epoch_nums==1:
    return EPOCHES_ALL
  elif epoch_nums==2:
    return EPOCHES_ALL_2
  elif epoch_nums==3:
    return EPOCHES_ALL_3


PRETRAINED_PATH = "./pre-trained"

OUTPUT_PATH = "./output"
TMP_PATH = "./output/tmp"

MODEL_EXTENSION = ".pth"