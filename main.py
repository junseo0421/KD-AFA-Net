import torch
import torch.utils.data
import torch.optim as optim
import torchvision.transforms
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
import os
from os.path import join, basename, splitext
from utils.loss import *
from models.Discriminator_ml import MsImageDis
from tensorboardX import SummaryWriter
from dataset import dataset_norm
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from loss import *
import timm

from utils.utils import *

from models.unet.sep_unet_model import *

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer, teacher_gen):
    gen.train()
    dis.train()

    projector_1.train()
    projector_2.train()
    projector_3.train()

    mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)
    ssim_loss = SSIM_loss().cuda(0)
    sobel_loss = Sobel_loss().cuda(0)

    acc_pixel_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    acc_total_sobel_loss = 0

    acc_original_kd_loss = 0
    acc_afa_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(train_loader):
            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

            iner_img = gt[:, :, :, 32:32 + 128]

            ## Generate Image
            I_pred, features_s = gen(mask_img)
            f1_s_afa, f2_s_afa, f3_s_afa = features_s["x1_out"], features_s["x2_out"], features_s["x3_out"]

            ## Compute losses
            ## Update Discriminator
            opt_dis.zero_grad()
            dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
            dis_loss = dis_adv_loss
            dis_loss.backward()
            opt_dis.step()

            for _ in range(2):
                I_pred, _ = gen(mask_img)

                mask_pred = I_pred[:, :, :, 32:32 + 128]

                # Pixel Reconstruction Loss
                pixel_rec_loss = mae(I_pred, gt) * 20

                # Texture Consistency Loss (IDMRF Loss)
                mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize

                # SSIM loss
                left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
                right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
                total_ssim_loss = left_loss + right_loss

                # Sobel loss
                sobel_left_loss = sobel_loss(I_pred[:, :, :, 0:32], gt[:, :, :, 0:32])
                sobel_right_loss = sobel_loss(I_pred[:, :, :, 160:192], gt[:, :, :, 160:192])
                total_sobel_loss = sobel_left_loss + sobel_right_loss

                sobel_loss_weight = 20.0

                total_sobel_loss = total_sobel_loss * sobel_loss_weight

                with torch.no_grad():
                    teacher_pred, features_t = teacher_gen(mask_img)
                    f1_t_afa, f2_t_afa, f3_t_afa = features_t["x1_out"], features_t["x2_out"], features_t["x3_out"]

                # original KD loss
                original_kd_loss = mae(teacher_pred, I_pred) * 20

                # AFA KD loss
                afa_loss = mse(projector_1(f1_s_afa), f1_t_afa) + mse(projector_2(f2_s_afa), f2_t_afa) + mse(
                    projector_3(f3_s_afa), f3_t_afa)

                afa_loss_weight = 10.0

                afa_loss = afa_loss * afa_loss_weight

                # Update Generator
                gen_adv_loss = dis.calc_gen_loss(I_pred, gt)

                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + afa_loss

                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_dis_adv_loss += dis_adv_loss.data
            acc_ssim_loss += total_ssim_loss
            acc_total_sobel_loss += total_sobel_loss

            acc_original_kd_loss += original_kd_loss.data
            acc_afa_loss += afa_loss.data

            total_gen_loss += gen_loss.data

            pbar.update(1)
            pbar.set_postfix({'gen_loss': gen_loss.item(),
                              'dis_loss': dis_loss.item(),
                              'sobel_loss': total_sobel_loss.item(),
                              'afa_loss': afa_loss.item()
                              })

    ## Tensor board
    writer.add_scalars('train/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/Sobel_loss',
                       {'sobel_loss': acc_total_sobel_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/afa_loss', {'afa_loss': acc_afa_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/total_gen_loss', {'total gen Loss': total_gen_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(train_loader.dataset)},
                       epoch)


def valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer, teacher_gen):
    gen.eval()
    dis.eval()

    projector_1.eval()
    projector_2.eval()
    projector_3.eval()

    mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)
    ssim_loss = SSIM_loss().cuda(0)
    sobel_loss = Sobel_loss().cuda(0)

    acc_pixel_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    acc_total_sobel_loss = 0

    acc_original_kd_loss = 0
    acc_afa_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(valid_loader), desc=f"Validation Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(valid_loader):
            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

            iner_img = gt[:, :, :, 32:32 + 128]

            with torch.no_grad():
                I_pred, features_s = gen(mask_img)
                f1_s_afa, f2_s_afa, f3_s_afa = features_s["x1_out"], features_s["x2_out"], features_s["x3_out"]

            mask_pred = I_pred[:, :, :, 32:32 + 128]

            # Update Discriminator
            opt_dis.zero_grad()
            dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
            dis_loss = dis_adv_loss

            # Pixel Reconstruction Loss
            pixel_rec_loss = mae(I_pred, gt) * 20

            # Texture Consistency Loss (IDMRF Loss)
            mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize

            # ## Update Generator

            # SSIM loss
            left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
            right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
            total_ssim_loss = left_loss + right_loss

            # Sobel loss
            sobel_left_loss = sobel_loss(I_pred[:, :, :, 0:32], gt[:, :, :, 0:32])
            sobel_right_loss = sobel_loss(I_pred[:, :, :, 160:192], gt[:, :, :, 160:192])
            total_sobel_loss = sobel_left_loss + sobel_right_loss

            sobel_loss_weight = 20.0

            total_sobel_loss = total_sobel_loss * sobel_loss_weight

            with torch.no_grad():
                teacher_pred, features_t = teacher_gen(mask_img)
                f1_t_afa, f2_t_afa, f3_t_afa = features_t["x1_out"], features_t["x2_out"], features_t["x3_out"]

            # original KD loss
            original_kd_loss = mae(teacher_pred, I_pred) * 20

            # AFA KD loss
            afa_loss = mse(projector_1(f1_s_afa), f1_t_afa) + mse(projector_2(f2_s_afa), f2_t_afa) + mse(
                projector_3(f3_s_afa), f3_t_afa)

            afa_loss_weight = 10.0

            afa_loss = afa_loss * afa_loss_weight

            gen_adv_loss = dis.calc_gen_loss(I_pred, gt)

            gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + afa_loss

            opt_gen.zero_grad()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_dis_adv_loss += dis_adv_loss.data
            acc_ssim_loss += total_ssim_loss.data
            acc_total_sobel_loss += total_sobel_loss.data

            acc_original_kd_loss += original_kd_loss.data
            acc_afa_loss += afa_loss.data

            total_gen_loss += gen_loss.data

            pbar.update(1)
            pbar.set_postfix(
                {'gen_loss': gen_loss.item(), 'dis_loss': dis_adv_loss.item(), 'sobel_loss': total_sobel_loss.item(), 'afa_loss': afa_loss.item()})

    ## Tensor board
    writer.add_scalars('valid/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/Sobel_loss',
                       {'sobel_loss': acc_total_sobel_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/afa_loss', {'afa_loss': acc_afa_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/total_gen_loss', {'total gen Loss': total_gen_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(valid_loader.dataset)},
                       epoch)

if __name__ == '__main__':

    seed_everything(2024)

    def get_args():

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str,
                            default='datasets', help='dataset directory')
        parser.add_argument('--save_dir', type=str,
                            default='Output', help='save directory')
        parser.add_argument('--name_dataset', type=str, choices=['HKdb-1', 'HKdb-2', 'SDdb-1', 'SDdb-2'],
                            default='HKdb-2', help='사용할 dataset')
        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=8)
        parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=600)
        parser.add_argument('--lr_G', type=float, help='generator learning rate', default=0.0004)
        parser.add_argument('--lr_D', type=float, help='discriminator learning rate', default=0.000004)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=bool, help='load pretrain weight', default=False)
        parser.add_argument('--test_flag', type=bool, help='testing while training', default=False)
        parser.add_argument('--adjoint', type=bool, help='if use adjoint in odenet', default=True)
        parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
                            default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
                            default=SAVE_WEIGHT_DIR)
        parser.add_argument('--log_dir', type=str, help='directory of saving logs', default=SAVE_LOG_DIR)
        parser.add_argument('--load_teacher_weight_dir', type=str, help='directory of teacher model weight',
                            default=LOAD_TEACHER_WEIGHT_DIR)

        opts = parser.parse_args()
        return opts

    args = get_args()

    name_dataset = args.name_dataset
    SAVE_BASE_DIR = args.save_dir

    SAVE_WEIGHT_DIR = join(SAVE_BASE_DIR, name_dataset, 'checkpoints')
    SAVE_LOG_DIR = join(SAVE_BASE_DIR, name_dataset, 'logs_all')
    LOAD_WEIGHT_DIR = join(SAVE_BASE_DIR, name_dataset, 'checkpoints')

    teacher_gen = AFA_Net(n_channels=3, n_classes=3).cuda()

    LOAD_TEACHER_WEIGHT_DIR = join('Output/teacher', f'{name_dataset}.pt')

    teacher_gen.load_state_dict(torch.load(LOAD_TEACHER_WEIGHT_DIR))
    print(f'Loading Teacher model weight...({name_dataset})')

    for param in teacher_gen.parameters():
        param.requires_grad = False

    teacher_gen.eval()

    base_dir = args.data_dir

    original_dir = join(base_dir, 'original_images_split', name_dataset)
    mask_dir = join(base_dir, 'mask_images_split_con', name_dataset)
    clahe_dir = join(base_dir, 'clahe_images_split', name_dataset)

    assert os.path.isdir(original_dir), f"Original directory does not exist: {original_dir}"
    assert os.path.isdir(mask_dir), f"Mask directory does not exist: {mask_dir}"
    assert os.path.isdir(clahe_dir), f"CLAHE directory does not exist: {clahe_dir}"

    original_list = glob(original_dir, '*', True)
    mask_list = glob(mask_dir, '*', True)
    clahe_list = glob(clahe_dir, '*', True)

    assert len(original_list) == len(mask_list) == len(clahe_list)

    print('Original list:', len(original_list))
    print('Mask list:', len(mask_list))
    print('CLAHE list:', len(clahe_list))

    train_ls_original, train_ls_mask, train_ls_clahe = [], [], []
    valid_ls_original, valid_ls_mask, valid_ls_clahe = [], [], []

    train_ls_original_list = original_list[:int(len(original_list) * 0.8)]
    train_ls_mask_list = mask_list[:int(len(mask_list) * 0.8)]
    train_ls_clahe_list = clahe_list[:int(len(clahe_list) * 0.8)]

    valid_ls_original_list = original_list[int(len(original_list) * 0.8):]
    valid_ls_mask_list = mask_list[int(len(mask_list) * 0.8):]
    valid_ls_clahe_list = clahe_list[int(len(clahe_list) * 0.8):]

    for path in train_ls_original_list:
        train_ls_original += glob(path, '*', True)

    for path in train_ls_mask_list:
        train_ls_mask += glob(path, '*', True)

    for path in train_ls_clahe_list:
        train_ls_clahe += glob(path, '*', True)

    for path in valid_ls_original_list:
        valid_ls_original += glob(path, '*', True)

    for path in valid_ls_mask_list:
        valid_ls_mask += glob(path, '*', True)

    for path in valid_ls_clahe_list:
        valid_ls_clahe += glob(path, '*', True)


    pred_step = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(join(args.log_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    print('Initializing model...')
    gen = LAFA_Net(n_channels=3, n_classes=3).cuda()  # student model

    dis = MsImageDis().cuda()

    projector_1 = Proj_1x1(in_channels=4, out_channels=32).cuda(0)
    projector_2 = Proj_1x1(in_channels=8, out_channels=64).cuda(0)
    projector_3 = Proj_1x1(in_channels=16, out_channels=128).cuda(0)

    opt_gen = optim.Adam(list(gen.parameters()) +
                         list(projector_1.parameters()) +
                         list(projector_2.parameters()) +
                         list(projector_3.parameters()), lr=args.lr_G, betas=(0.5, 0.999), weight_decay=1e-4)

    opt_dis = optim.Adam(dis.parameters(), lr=args.lr_D, betas=(0.5, 0.999), weight_decay=1e-4)


    # Load pre-trained weight
    if args.load_pretrain:
        start_epoch = 500
        print(f'Loading model weight...at epoch {start_epoch}')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, f'Gen_former_{start_epoch}.pt')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, f'Dis_former_{start_epoch}.pt')))
    else:
        start_epoch = 0

    # Load data
    print('Loading data...')

    transformations = transforms.Compose(
        [torchvision.transforms.RandomResizedCrop((192, 192), scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), ),
         CenterCrop(192), ToTensor(), Normalize(mean, std)])  # augmentation
    transformations_valid = transforms.Compose(
        [torchvision.transforms.Resize((192, 192)), ToTensor(), Normalize(mean, std)])

    train_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128,
                              imglist1=train_ls_original,
                              imglist2=train_ls_mask,
                              imglist3=train_ls_clahe)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('train data: %d images' % (len(train_loader.dataset)))

    valid_data = dataset_norm(root=args.train_data_dir, transforms=transformations_valid, imgSize=192, inputsize=128,
                              imglist1=valid_ls_original,
                              imglist2=valid_ls_mask,
                              imglist3=valid_ls_clahe)
    valid_loader = DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
    print('valid data: %d images' % (len(valid_loader.dataset)))

    # Train & test the model
    for epoch in range(start_epoch + 1, 1 + args.epochs):
        print("----Start training[%d / %d]----" % (epoch, args.epochs))

        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer, teacher_gen)
        valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer, teacher_gen)

        if (epoch % 10) == 0:
            torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_former_%d.pt' % epoch))
            torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_former_%d.pt' % epoch))

    writer.close()
