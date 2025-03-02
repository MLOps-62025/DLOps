import os
import argparse
from get_data import get_data, read_params

################  CREATING FOLDER - STAR ########################

def create_fold(config, image=None):
    config  = get_data(config)
    dirr = config['load_data']['preprocessed_data']
    cla = config['load_data']['num_classes']
    #print(dirr)
    #print(cla)
    if os.path.exists(dirr+'/'+'train'+'/'+'class_0') and os.path.exists(dirr+'/'+'test'+'/'+'class_0'):
        print("Training and test folder already exists.....!")
        print("I am skipping it....!")
    else:
        os.mkdir(dirr+'/'+'train')
        os.mkdir(dirr+'/'+'test')
        for i in range(cla):
            os.mkdir(dirr+'/'+'train'+'/'+'class_'+str(i))
            os.mkdir(dirr+'/'+'test'+'/'+'class_'+str(i))
        print("Folder is successfully create....!!")

######################### CREATING FOLDER - END ##############################

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passes_args= args.parse_args()
    a = create_fold(config=passes_args.config)