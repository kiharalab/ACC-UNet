"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle
from nets.Unet22_2_3_2_grp import Unet22_2_3_2_grp
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

from nets.ACC_UNet import ACC_UNet
from nets.UCTransNet import UCTransNet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet

import json
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))


    return dice_pred, iou_pred
    

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    input_img.to('cpu')


    input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()
    labs = labs[0]
    output = output[0,0,:,:].cpu().detach().numpy()

    if(True):
        pickle.dump({
            'input':input_img,
            'output':(output>=0.5)*1.0,            
            'ground_truth':labs,
            'dice':dice_pred_tmp,
            'iou':iou_tmp
        },
        open(vis_save_path+'.p','wb'))

    if(False):
        
        plt.figure(figsize=(10,3.3))
        plt.subplot(1,3,1)
        plt.imshow(input_img)
        plt.subplot(1,3,2)
        plt.imshow(labs,cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow((output>=0.5)*1.0,cmap='gray')    
        plt.suptitle(f'Dice score : {np.round(dice_pred_tmp,3)}\nIoU : {np.round(iou_tmp,3)}')
        plt.tight_layout()
        plt.savefig(vis_save_path)
        plt.close()


    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name =="GlaS_exp1":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="GlaS_exp2":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="GlaS_exp3":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    
    elif config.task_name =="ISIC18_exp1":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="ISIC18_exp2":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="ISIC18_exp3":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    
    
    elif config.task_name =="Clinic_exp1":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Clinic_exp2":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Clinic_exp3":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    

    elif config.task_name =="BUSI_exp1":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="BUSI_exp2":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="BUSI_exp3":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    

    elif config.task_name =="Covid_exp1":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Covid_exp2":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Covid_exp3":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"


    save_path  = config.task_name +'/'+ config.model_name +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path+'test.result','a')
    fp.write(str(datetime.now())+'\n')

    
    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()   
        model = ACC_UNet(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=config.n_filts)

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        config_vit = config.get_CTranS_config()   
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)
        
    elif model_type == 'SwinUnet':            
        model = SwinUnet()
        model.load_from()

    elif model_type == 'SMESwinUnet':            
        model = SMESwinUnet(n_channels=config.n_channels,n_classes=config.n_labels)
        model.load_from()

    elif model_type.split('_')[0] == 'MultiResUnet1':          
        model = MultiResUnet(n_channels=config.n_channels,n_classes=config.n_labels,nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))
    
    
    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            ###########fig, ax = plt.subplots()
            ###########plt.imshow(img_lab, cmap='gray')
            ###########plt.axis("off")
            height, width = config.img_size, config.img_size
            ###########fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            ###########plt.gca().xaxis.set_major_locator(plt.NullLocator())
            ###########plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ###########plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ###########plt.margins(0, 0)
            ###########plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            ###########plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i)+'.png',
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    
    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()



