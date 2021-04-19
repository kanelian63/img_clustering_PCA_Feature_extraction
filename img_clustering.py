import os, glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
#matplotlib.use('Agg')

import torch
import torchvision.models as models

from torch.utils.data import Dataset
from torchvision import transforms
torch.manual_seed(17)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.device(0)
cuda = torch.device('cuda')

img_dir = './face_cc/face_src_temp'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
print(len(img_list))
labels = pd.DataFrame(index=range(0, len(img_list)), columns=['limits_velocity'])
labels['limits_velocity'] = 1

transforms = transforms.Compose([transforms.Resize((128, 128)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class CustomDataset(Dataset):

    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels['limits_velocity'].values
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_name = self.img_dir[idx]
        image = Image.open(img_name)
        label = self.labels[idx]
        label = torch.from_numpy(np.array(label))
        transformed_image = transforms(image)

        return transformed_image, label


clustering_dataset = CustomDataset(img_list, labels, transforms)
print(len(clustering_dataset))

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.vgg16(pretrained=True).cuda()
#model = models.resnet18(pretrained=True).cuda()
#model = models.resnet101(pretrained=True).cuda()
vgg_pretrained_features = model.features

which_layer_to_visualize = 4 # Select visualization layer
print(vgg_pretrained_features[which_layer_to_visualize])

data = {}

with torch.no_grad():
    for idx, (img, label) in enumerate(clustering_dataset):
        print(f'{idx}번째 이미지의 feature 추출중..')
        handle = vgg_pretrained_features[which_layer_to_visualize].register_forward_hook(
            get_activation(str(which_layer_to_visualize)))

        target_img = img[None, ...].cuda()
        out = vgg_pretrained_features(target_img)
        handle.remove()

        #act = activation[str(which_layer_to_visualize)].squeeze().cpu()
        #act = (act * 255).numpy().astype(np.uint8)
        #print(idx, act.shape)
        #out = model(target_img)
        print(out.shape)
        data[idx] = out.cpu().numpy()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

feat = np.array(list(data.values()))
print(f'feature shape는 {feat.shape}입니다.')
feat = feat.squeeze()
n, c, w, h = feat.shape
feat = feat.reshape(-1, c*w*h)
print(f'feature shape는 {feat.shape}입니다.')

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=22)
kmeans.fit(x)
print(f'clustering한 객체의 수는 {len(kmeans.labels_)}입니다.')

# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize = (25, 25))

    # gets the list of filenames for a cluster
    files = groups[cluster]

    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]

    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(15, 15, index+1);
        img = Image.open(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

groups = {}

for idx in set(kmeans.labels_):
    groups[idx] = []

for file, cluster in zip(img_list, kmeans.labels_):
    groups[cluster].append(file)
#view_cluster(0)

import shutil
dst_dir = './face_cc/clustering'
for i in range(n_clusters):
    os.mkdir(os.path.join(dst_dir, str(i)))
    for img_file in groups[i]:
        nm = img_file.split('/')[-1]
        shutil.copy(img_file, os.path.join(dst_dir, str(i), nm))

