import os
import torch
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from net.cspLoss import CspLoss
from load_data.dataloader import PETS2009
from load_data.load_data import get_pets2009
from torch.utils.data import DataLoader
from net.csp import *
from config import Config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

load_ckpt = 9

parser = argparse.ArgumentParser(description='CSP Training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
if load_ckpt > 0:
    parser.add_argument('--resume_net', action="store_true", default=True, help='resume')
    parser.add_argument('--resume_epoch', default=load_ckpt, type=int, help='resume iter for retraining')
else:
    parser.add_argument('--resume_net', action="store_true", default=False, help='resume')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
args = parser.parse_args()

num_gpu = args.ngpu
num_workers = args.num_workers
resume_epoch = args.resume_epoch


def validation(net, validationloader, device, criterion):
    net.eval()
    total_center_loss = 0.0
    total_reg_loss_h = 0.0
    total_reg_loss_w = 0.0
    total_offset_loss = 0.0

    count = 0
    for i, data in enumerate(validationloader, 0):
        tensor_img_input, tensor_center_gt_input, tensor_scale_gt_input, tensor_offset_gt_input = data
        tensor_img_input = tensor_img_input.to(device)
        tensor_center_gt_input = tensor_center_gt_input.to(device)
        tensor_scale_gt_input = tensor_scale_gt_input.to(device)
        tensor_offset_gt_input = tensor_offset_gt_input.to(device)
        output = net(tensor_img_input)
        center_loss, reg_loss_h, reg_loss_w, offset_loss = criterion(output, tensor_center_gt_input,
                                                                     tensor_scale_gt_input,
                                                                     tensor_offset_gt_input)
        count += 1
        total_center_loss += round(center_loss.item(), 5)
        total_reg_loss_h += round(reg_loss_h.item(), 5)
        total_reg_loss_w += round(reg_loss_w.item(), 5)
        total_offset_loss += round(offset_loss.item(), 5)
    '''
     print('====    Validation loss: center_loss:{}   reg_loss_h:{}   reg_loss_w:{}   offset_loss:{}'.format
          (total_center_loss / count, total_reg_loss_h / count, total_reg_loss_w / count, total_offset_loss / count))
    '''
    print('validation, {}, {}, {}, {}, {}'.format(0, round(center_loss.item(), 5), round(reg_loss_h.item(), 5),
                                                round(reg_loss_w.item(), 5), round(offset_loss.item(), 5)))

def train():
    config = Config()
    train_img, train_label, train_mean, test_img, test_label, test_mean = get_pets2009(dir='./data_PETS2009',
                                                                                       pkl_dir='./data_cache')

    traindataset = PETS2009(train_img, train_label, train_mean, type='train', config=config)
    trainloader = DataLoader(traindataset, batch_size=4, num_workers=num_workers, shuffle=True)
    validationdataset = PETS2009(test_img, test_label, test_mean, type='train', config=config)
    validationloader = DataLoader(validationdataset, batch_size=4, num_workers=num_workers, shuffle=True)

    net = Csp('train')
    device = torch.device('cuda')
    net.to(device)

    if args.resume_net:
        pretrained_path = './weights/CSP_epoch_' + str(resume_epoch) + '.pth'
        print('Loading pretrained model from {}'.format(pretrained_path))
        model = torch.load(pretrained_path, map_location=device)
        net.load_state_dict(model)

    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))

    cudnn.benchmark = True
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = CspLoss()

    isAdjust_lr = False
    validation(net, validationloader, device, criterion)
    for epoch in range(resume_epoch + 1, config.epoch_size + 1):
        net.train()
        if epoch > 200 and not isAdjust_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
            isAdjust_lr = True

        for i, data in enumerate(trainloader, 0):
            tensor_img_input, tensor_center_gt_input, tensor_scale_gt_input, tensor_offset_gt_input = data
            tensor_img_input = tensor_img_input.to(device)
            tensor_center_gt_input = tensor_center_gt_input.to(device)
            tensor_scale_gt_input = tensor_scale_gt_input.to(device)
            tensor_offset_gt_input = tensor_offset_gt_input.to(device)

            output = net(tensor_img_input)

            optimizer.zero_grad()
            center_loss, reg_loss_h, reg_loss_w, offset_loss = criterion(output, tensor_center_gt_input,
                                                                         tensor_scale_gt_input, tensor_offset_gt_input)
            loss = 0.01 * center_loss + 1 * reg_loss_h + 1 * reg_loss_w + 0.1 * offset_loss
            loss.backward()
            optimizer.step()
            '''
            if i % 50 == 0:
                print('epoch:{}/{}  i:{}  center_loss:{}   reg_loss_h:{}   reg_loss_w:{}   offset_loss:{}'.format
                      (epoch, config.epoch_size, i, round(center_loss.item(), 5), round(reg_loss_h.item(), 5),
                       round(reg_loss_w.item(), 5), round(offset_loss.item(), 5)))
           '''
        '''
        print(
            'Training loss: epoch:{}/{}   center_loss:{}   reg_loss_h:{}   reg_loss_w:{}   offset_loss:{}'.format
            (epoch, config.epoch_size, round(center_loss.item(), 5), round(reg_loss_h.item(), 5),
             round(reg_loss_w.item(), 5), round(offset_loss.item(), 5)))
        '''
        print('training, {}, {}, {}, {}, {}'.format(epoch, round(center_loss.item(), 5), round(reg_loss_h.item(), 5), round(reg_loss_w.item(), 5), round(offset_loss.item(), 5)))
        if (epoch % 3 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
            torch.save(net.module.state_dict(), './weights/' + 'CSP_epoch_' + str(epoch) + '.pth')
            validation(net, validationloader, device, criterion)


if __name__ == '__main__':
    train()
