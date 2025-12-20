import os
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
import random
from model.OMANet import OMANet  

from dataset_omanet import get_loader, test_dataset_snr

from utils_ch.loss_omanet import bce_iou_loss, AdaptiveSegLoss
from utils_ch.metrics import calculate_metrics
from torchvision.utils import save_image
import logging
import torch.nn as nn

import warnings
warnings.filterwarnings(action='ignore')

 
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

class Trainer:
    def __init__(self, opt, model):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(opt.save_path, exist_ok=True)

        self.model = model.to(self.device)
        self.num_epoch = opt.num_epoch
        self.start_epoch = 0
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(opt.lr), weight_decay=1e-5)
        print(self.device)
        print(self.optimizer)

        # data load
        if opt.snr_option:
            self.trainloader = get_loader(opt.train_root, opt.batch_size, opt.train_size, opt.snr_option, shuffle=True, drop_last=True)
            dataset = test_dataset_snr(opt.valid_root, opt.train_size)
            self.valloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)
        else:
            self.trainloader = get_loader(opt.train_root, opt.batch_size, opt.train_size, opt.snr_option, shuffle=True, drop_last=True)
            dataset = test_dataset_snr(opt.valid_root, opt.train_size)
            self.valloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

        # directory settings
        self.ckpoint_path = os.path.join(opt.save_path, 'checkpoint')
        self.images_path = os.path.join(opt.save_path, 'images')
        self.log_path = os.path.join(opt.save_path, 'log')

        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.ckpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        self.best_miou = 0

        # logging
        self.logging_init(opt)

        if opt.resume_option:
            print('############# resume training #############')
            checkpoint_path = os.path.join(self.ckpoint_path, 'latest.pth')
            self.start_epoch, optimizer_statedict, self.best_miou = self.load_checkpoint(checkpoint_path)
            self.optimizer.load_state_dict(optimizer_statedict)
            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info('-----------------------------------------------------------------------------')
            logging.info(f'Resume Epoch [{self.start_epoch}] - current_lr: {current_lr:.6f}, Best miou : {self.best_miou:.4f}')
            logging.info('-----------------------------------------------------------------------------')

        else:
            logging.info('>>> Not resuming. Start new training.')
            
    def logging_init(self, opt):
        """
        Logging 및 file handler 설정
        """
        log_file = os.path.join(self.log_path, 'log.log')
        logging.basicConfig(
            filename=log_file,
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO,
            filemode='a',
            datefmt='%Y-%m-%d %I:%M:%S %p'
        )
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        logging.info(">>> current mode: network-train/val")
        logging.info('>>> config: {}'.format(opt))

    def save_checkpoint(self, epoch, filename):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({
            'network': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'best_miou': self.best_miou
        }, filename)

    def load_checkpoint(self, weights_path):
        chkpoint = torch.load(weights_path)
        self.model.load_state_dict(chkpoint['network'])
        return chkpoint['epoch'], chkpoint['optimizer'],  chkpoint['best_miou']
    

    def compute_loss(self, predictions, snr, targets, epoch, opt):
    
        object_prob, P4, P3, P2, P1, D3, D2, D1, aux = predictions
    
        gts = targets
        
        self.loss4s = bce_iou_loss(P4, gts) 
        self.loss3s = bce_iou_loss(P3, gts)
        self.loss2s = bce_iou_loss(P2, gts)
        self.loss1s = bce_iou_loss(P1, gts)

        self.loss3d = bce_iou_loss(D3, gts)
        self.loss2d = bce_iou_loss(D2, gts)
        self.loss1d = bce_iou_loss(D1, gts)

        loss_snr = AdaptiveSegLoss()
        self.losst, self.losss, self.losso = loss_snr(aux, gts, snr, object_prob, epoch, opt)
        
        self.seg_loss = self.loss2s + self.loss3s + self.loss4s + self.loss1s
        self.seg_loss2 = self.loss1d + self.loss2d + self.loss3d 
        
        self.total_loss = 0.5*self.seg_loss + self.seg_loss2 + 2*self.losst 

        return self.total_loss

    def log_losses(self, epoch, phase, avg_losses, summary):
       
        logging.info(
            f"{phase} Epoch [{epoch}] - "
            f"Loss_L4: {avg_losses['loss4s']:.4f}, "
            f"Loss_L3: {avg_losses['loss3s']:.4f}, "
            f"Loss_L2: {avg_losses['loss2s']:.4f}, "
            f"Loss_L2: {avg_losses['loss1s']:.4f}, "
            f"Loss_D3: {avg_losses['loss3d']:.4f}, "
            f"Loss_D2: {avg_losses['loss2d']:.4f}, "
            f"Loss_D1: {avg_losses['loss1d']:.4f}, "
            f"Loss_aux: {avg_losses['aux_loss']:.4f}, "
            f"Loss_aux_s: {avg_losses['aux_s_loss']:.4f}, "
            f"Loss_aux_o: {avg_losses['aux_o_loss']:.4f}, "
            f"Total_Loss: {avg_losses['total_loss']:.4f}"
        )

        summary.add_scalar(f'{phase}/Loss_Structure_L4', avg_losses['loss4s'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_L3', avg_losses['loss3s'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_L2', avg_losses['loss2s'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_L1', avg_losses['loss1s'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_D3', avg_losses['loss3d'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_D2', avg_losses['loss2d'], epoch)
        summary.add_scalar(f'{phase}/Loss_Structure_D1', avg_losses['loss1d'], epoch)
        summary.add_scalar(f'{phase}/Loss_aux', avg_losses['aux_loss'], epoch)
        summary.add_scalar(f'{phase}/Loss_aux_s', avg_losses['aux_s_loss'], epoch)
        summary.add_scalar(f'{phase}/Loss_aux_o', avg_losses['aux_o_loss'], epoch)
        
        summary.add_scalar(f'{phase}/Loss_Total', avg_losses['total_loss'], epoch)


    def Train_snr(self):
        model = self.model.to(self.device)
        summary = SummaryWriter(f'{self.log_path}/tensorboard')  
        best_score = {'epoch': self.start_epoch, 'miou': self.best_miou, 'loss': 0}
        
        try:
            for epoch in range(self.start_epoch, self.num_epoch):
                tr_obj_iou_score = []
                te_obj_iou_score = []
                tr_miou_score = []
                te_miou_score = []
                tr_loss = 0
                te_loss = 0

                loss_components_train = {
                    'loss4s': 0, 'loss3s': 0, 'loss2s': 0,'loss1s': 0,
                    'loss3d': 0, 'loss2d': 0, 'loss1d': 0,
                    'exposure_loss': 0, 'aux_loss': 0, 
                    'aux_s_loss': 0, 'aux_o_loss': 0, 
                    'total_loss': 0
                }

                model.train()
                for idx, (images, gts, snrs) in enumerate(tqdm(self.trainloader, desc=f'[Train {epoch}/{self.num_epoch}]')):
                    images, gts, snrs = images.to(self.device), gts.to(self.device), snrs.to(self.device)


                    self.optimizer.zero_grad()
                    predictions, outs = model(images, epoch)
                    loss = self.compute_loss(predictions, snrs, gts, epoch, 'tr')
                    loss.backward()
                    self.optimizer.step()

                    loss_components_train['loss4s'] += self.loss4s.item()
                    loss_components_train['loss3s'] += self.loss3s.item()
                    loss_components_train['loss2s'] += self.loss2s.item()
                    loss_components_train['loss1s'] += self.loss1s.item()
                    loss_components_train['loss3d'] += self.loss3d.item()
                    loss_components_train['loss2d'] += self.loss2d.item()
                    loss_components_train['loss1d'] += self.loss1d.item()
                    loss_components_train['aux_loss'] += self.losst.item()
                    loss_components_train['aux_s_loss'] += self.losss.item()
                    loss_components_train['aux_o_loss'] += self.losso.item()
                    loss_components_train['total_loss'] += loss.item()

                    results = calculate_metrics(gts, predictions[-2])  
                    tr_obj_iou_score.append(torch.tensor(results['object_iou']))
                    tr_miou_score.append(torch.tensor(results['miou']))
                    tr_loss += loss.item()
                    
                avg_train_loss = tr_loss / len(self.trainloader)
                avg_train_obj_iou = torch.stack(tr_obj_iou_score).mean().item() if tr_obj_iou_score else 0
                avg_train_miou = torch.stack(tr_miou_score).mean().item() if tr_miou_score else 0

                avg_loss_components_train = {key: val / len(self.trainloader) for key, val in loss_components_train.items()}


                # validation
                loss_components_valid = {
                    'loss4s': 0, 'loss3s': 0, 'loss2s': 0, 'loss1s': 0,
                    'loss3d': 0, 'loss2d': 0, 'loss1d': 0,
                    'exposure_loss': 0, 'aux_loss': 0, 
                    'aux_s_loss': 0, 'aux_o_loss': 0, 
                    'total_loss': 0
                }

                with torch.no_grad():
                    model.eval()
                    te_loss = 0
                    te_obj_iou_score = []
                    te_miou_score = []

                    for idx, (image, gt, snr, _) in enumerate(tqdm(self.valloader, desc=f'[Valid {epoch}/{self.num_epoch}]')):
                        image, gt, snr = image.to(self.device), gt.to(self.device), snr.to(self.device)

                        prediction, out = model(image, epoch)
                        loss = self.compute_loss(prediction, snr, gt, epoch, 'val')
                        
                        loss_components_valid['loss4s'] += self.loss4s.item()
                        loss_components_valid['loss3s'] += self.loss3s.item()
                        loss_components_valid['loss2s'] += self.loss2s.item()
                        loss_components_valid['loss1s'] += self.loss1s.item()
                        loss_components_valid['loss3d'] += self.loss3d.item()
                        loss_components_valid['loss2d'] += self.loss2d.item()
                        loss_components_valid['loss1d'] += self.loss1d.item()
                        loss_components_valid['aux_loss'] += self.losst.item()
                        loss_components_valid['aux_s_loss'] += self.losss.item()
                        loss_components_valid['aux_o_loss'] += self.losso.item()
                        loss_components_valid['total_loss'] += loss.item()

                        result = calculate_metrics(gt, prediction[-2])
                        te_obj_iou_score.append(torch.tensor(result['object_iou']))
                        te_miou_score.append(torch.tensor(result['miou']))
                        te_loss += loss.item()

                        if epoch % 50 == 0:
                            self.save_concat_results3(image, gt, gt, prediction[1], snr, out[0], prediction[-1], prediction[-2], self.images_path, epoch, idx, normalize=True)
                          

                    avg_valid_loss = te_loss / len(self.valloader)
                    avg_valid_obj_iou = torch.stack(te_obj_iou_score).mean().item() if te_obj_iou_score else 0
                    avg_valid_miou = torch.stack(te_miou_score).mean().item() if te_miou_score else 0

                    avg_loss_components_valid = {key: val / len(self.valloader) for key, val in loss_components_valid.items()}

                if best_score['miou'] <= avg_valid_miou:
                    best_score['epoch'] = epoch
                    best_score['miou'] = avg_valid_miou
                    best_score['loss'] = avg_valid_loss
                    self.best_miou = avg_valid_miou
                    self.save_checkpoint(epoch, 'best_miou.pth')

                logging.info('-----------------------------------------------------------------------------')
                logging.info(
                    f'Epoch [{epoch}/{self.num_epoch}] - '
                    f'Train Loss: {avg_train_loss:.4f}, Train Obj IOU: {avg_train_obj_iou:.4f}, Train mIoU: {avg_train_miou:.4f}'
                )
                logging.info(
                    f'Epoch [{epoch}/{self.num_epoch}] - '
                    f'Valid Loss: {avg_valid_loss:.4f}, Valid Obj IOU: {avg_valid_obj_iou:.4f}, Valid mIoU: {avg_valid_miou:.4f}'
                )
                logging.info(
                    f"Best miou epoch: {best_score['epoch']}, Best miou: {best_score['miou']:.4f}"
                )
                logging.info('-----------------------------------------------------------------------------')

                self.log_losses(epoch, 'Train', avg_loss_components_train, summary)
                self.log_losses(epoch, 'Valid', avg_loss_components_valid, summary)
                
                summary.add_scalar('train/miou', avg_train_miou, epoch)
                summary.add_scalar('train/obj_iou', avg_train_obj_iou, epoch)
                summary.add_scalar('train/loss', avg_train_loss, epoch)

                summary.add_scalar('valid/miou', avg_valid_miou, epoch)
                summary.add_scalar('valid/obj_iou', avg_valid_obj_iou, epoch)
                summary.add_scalar('valid/loss', avg_valid_loss, epoch)

                summary.flush()

                self.save_checkpoint(epoch, 'latest.pth')
                self.save_checkpoint(epoch, f'epoch_{epoch}.pth')

        except KeyboardInterrupt:
            print('Keyboard Interrupt: save model and exit.')
            self.save_checkpoint(epoch, f'epoch_{epoch + 1}.pth')
            raise
        finally:
            summary.close()
    
    
    def save_concat_results3(self, image: torch.Tensor, gt: torch.Tensor, snr: torch.Tensor, v_ori: torch.Tensor, v_norm: torch.Tensor,
                         image_n: torch.Tensor, coarse2: torch.Tensor, pred: torch.Tensor, save_dir: str, epoch: str, filename: str, normalize: bool = True) -> None:

        os.makedirs(save_dir, exist_ok=True)

        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)

            image_n = image_n * std + mean
            image_n = torch.clamp(image_n, 0, 1)

        pred = (torch.sigmoid(pred) > 0.5).float()
        coarse2 = coarse2.float()
        gt = (gt > 0.5).float()
        snr = (torch.sigmoid(snr) > 0.5).float()

        batch_size = gt.size(0)
        gt_3ch = gt.repeat(1, 3, 1, 1)
        snr_3ch = snr.repeat(1, 3, 1, 1)
        v_ori_3ch = v_ori.repeat(1, 3, 1, 1)
        v_norm_3ch = v_norm.repeat(1, 3, 1, 1)
        coarse2_3ch = coarse2.repeat(1, 3, 1, 1)
        pred_3ch = pred.repeat(1, 3, 1, 1)

        row1 = torch.cat([image, v_ori_3ch, v_norm_3ch, image_n], dim=3)
        row2 = torch.cat([gt_3ch, snr_3ch, coarse2_3ch, pred_3ch], dim=3)
        concat_tensor = torch.cat([row1, row2], dim=2)

        for i in range(batch_size):
            save_path = os.path.join(save_dir, f"epoch_{epoch}_{filename}.png")

            save_image(
                concat_tensor[i],
                save_path,
                normalize=False,
                padding=0
            )
    




if __name__ == '__main__':
    seed_everything(2024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='D:/marine_data/MAS3K_low_n/MAS3K/fold1/train/') 
    parser.add_argument('--valid_root', type=str, default='D:/marine_data/MAS3K_low_n/MAS3K/fold1/valid/')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--num_epoch', type=int, default=300, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--train_size', type=int, default=352, help='training dataset size')
    parser.add_argument('--snr_option', type=bool, default=True, help='training aux option')
    parser.add_argument('--resume_option', type=bool, default=False, help='resume checkpoint option')
    parser.add_argument('--save_path', type=str,
                        default='B:/CODE/SEC_pro/checkpoints/MAS3K_f1/OMANet',
                        help='save path')
    
    opt = parser.parse_args()

    encoder_backbone = 'B:/CODE/SEC_pro/model/encoder/pvt_v2_b2.pth'
    model = OMANet(load_path=encoder_backbone).cuda()
    
    trainer = Trainer(opt, model)

    trainer.Train_snr()
