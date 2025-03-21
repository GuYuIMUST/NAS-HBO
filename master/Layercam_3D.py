import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import transforms as transforms
from dataloader import lunanod
import pandas as pd
import matplotlib.animation as animation  # liu github
import matplotlib.pyplot as plt  # liu add
from torchcam.methods import SmoothGradCAMpp, LayerCAM  # guyu 新增LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from scipy.ndimage.interpolation import zoom
import cv2
import argparse



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--pretrain_model_path', type=str, default=r"D:\wangmansheng\92.39130235754925_217_model.pt", help='pretrain_model_path')
args = parser.parse_args()


# the test model
def load_data(test_data_path, preprocess_path, fold, batch_size, num_workers):
    crop_size = 32
    black_list = []

    pix_value, npix = 0, 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list:
                continue
            data = np.load(os.path.join(preprocess_path, file_name))
            pix_value += np.sum(data)
            npix += np.prod(data.shape)
    pix_mean = pix_value / float(npix)
    pix_value = 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list: continue
            data = np.load(os.path.join(preprocess_path, file_name)) - pix_mean
            pix_value += np.sum(data * data)
    pix_std = np.sqrt(pix_value / float(npix))
    print(f'pix_mean, pix_std: {pix_mean}, {pix_std}')
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),
    ])

    # load data list
    test_file_name_list = []  # this will be used later in the code
    test_label_list = []
    test_feat_list = []

    data_frame = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv',
                             names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

    all_list = data_frame['seriesuid'].tolist()[1:]
    label_list = data_frame['malignant'].tolist()[1:]
    crdx_list = data_frame['coordX'].tolist()[1:]
    crdy_list = data_frame['coordY'].tolist()[1:]
    crdz_list = data_frame['coordZ'].tolist()[1:]
    dim_list = data_frame['diameter_mm'].tolist()[1:]
    # test id
    test_id_list = []
    for file_name in os.listdir(test_data_path + str(fold) + '/'):

        if file_name.endswith('.mhd'):
            test_id_list.append(file_name[:-4])
    mxx = mxy = mxz = mxd = 0
    for srsid, label, x, y, z, d in zip(all_list, label_list, crdx_list, crdy_list, crdz_list, dim_list):
        mxx = max(abs(float(x)), mxx)
        mxy = max(abs(float(y)), mxy)
        mxz = max(abs(float(z)), mxz)
        mxd = max(abs(float(d)), mxd)
        if srsid in black_list:
            continue
        # crop raw pixel as feature
        data = np.load(os.path.join(preprocess_path, srsid + '.npy'))
        bgx = int(data.shape[0] / 2 - crop_size / 2)
        bgy = int(data.shape[1] / 2 - crop_size / 2)
        bgz = int(data.shape[2] / 2 - crop_size / 2)
        data = np.array(data[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size])
        y, x, z = np.ogrid[-crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2]
        mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
        feat = np.zeros((crop_size, crop_size, crop_size), dtype=float)
        feat[mask] = 1
        if srsid.split('-')[0] in test_id_list:
            test_file_name_list.append(srsid + '.npy')
            test_label_list.append(int(label))
            test_feat_list.append(feat)
    for idx in range(len(test_feat_list)):
        test_feat_list[idx][-1] /= mxd

    test_set = lunanod(preprocess_path, test_file_name_list, test_label_list, test_feat_list, train=False,
                       download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def cam_visualization_single_cube_fusecam(model, test_loader):
    model.eval()
    for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        cam_extractor_all = LayerCAM(model, target_layer=[
            'cells.19._ops.0',
            'cells.19._ops.1',
            'cells.19._ops.2',
            'cells.19._ops.3',
            'cells.19._ops.4',
            'cells.19._ops.5',
            'cells.19._ops.6',
            'cells.19._ops.7'
        ], input_shape=(32, 32, 32))

        # cam_extractor_all = LayerCAM(model, target_layer=['cells.19'], input_shape=(32, 32, 32))

        out, _ = model(inputs)
        class_idx = out.squeeze(0).squeeze(0).argmax().item()
        cam_layer_all = cam_extractor_all(class_idx, out)

        fused_cam = cam_extractor_all.fuse_cams(cam_layer_all)
        activation_map_numpy = fused_cam.cpu().numpy()
        np.set_printoptions(linewidth=150)
        print(f'activation_map_numpy_fuse_cam:\n{activation_map_numpy}')
        print(f'activation_map_numpy_fuse_cam.shape:{activation_map_numpy.shape}')

        # add squeeze
        inputs_new = inputs.squeeze()
        print(f'inputs_new.shape:{inputs_new.shape}')
        inputs_new_numpy = inputs_new.cpu().numpy()
        inputs_new_numpy_norm = normalization(inputs_new_numpy)

        resize_factor = inputs_new_numpy.shape[0] / activation_map_numpy[0].shape[0]
        activation_map_numpy = activation_map_numpy.reshape(8, 8, 8)
        activation_map_numpy_resized = zoom(activation_map_numpy, resize_factor, mode='nearest', order=2)

        for idx in range(inputs_new_numpy.shape[0]):
            inputs_new_norm_slice = inputs_new_numpy_norm[idx, :, :]
            activation_map_numpy_slice = activation_map_numpy_resized[idx, :, :]

            inputs_new_norm_slice_original = cv2.cvtColor(inputs_new_norm_slice, cv2.COLOR_GRAY2RGB)
            inputs_new_norm_slice_original = np.uint8(255 * inputs_new_norm_slice_original)  # 把范围从0-1转为0-255
            # 指定保存目录
            save_dir = r'D:\wangmansheng\code\CAM\result3/'
            save_dir_2 = r'D:\wangmansheng\code\CAM\result4/'
            file_name_1 = f'cube_{idx}_original.jpg'
            file_name_2 = f'cube_{idx}_with_fuse_cam.jpg'
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_dir_2, exist_ok=True)
            file_path_1 = os.path.join(save_dir, file_name_1)
            file_path_2 = os.path.join(save_dir_2, file_name_2)

            cv2.imwrite(file_path_1, inputs_new_norm_slice_original)

            cam_image = show_cam_on_image(inputs_new_norm_slice, activation_map_numpy_slice, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(file_path_2, cam_image)

        print('finish')


if __name__ == '__main__':
    fold = 5
    batch_size = 1
    num_workers = 0
    test_data_path = r'D:\wangmansheng\wangmansheng_2\code\CAM\data\subset'
    preprocess_path = r'D:\wangmansheng\wangmansheng_2\code\CAM\data\crop_v3'
    pretrain_model_path = args.pretrain_model_path
    checkpoint = torch.load(pretrain_model_path)
    model = checkpoint['model']
    model = model.cuda()
    epoch = checkpoint['epoch']
    valid_acc = checkpoint['valid_acc']
    print(model)
    print(dict(model.named_modules()).keys())
    test_data_loader = load_data(test_data_path, preprocess_path, fold, batch_size, num_workers)
    cam_visualization_single_cube_fusecam(model, test_data_loader)


