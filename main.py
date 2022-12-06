from dataloaders.pc_dataset import SemanticKITTI
import argparse
import yaml
from easydict import EasyDict
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from torch.utils.data import DataLoader
from depth_completion._2dpapenet import get_model as DepthCompletionModel
import tensorboardX

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

args = argparse.ArgumentParser()
args = args.parse_args(args=[])
with open('./config/semantic.yaml') as stream:
    config = yaml.safe_load(stream)
config.update(vars(args))
args = EasyDict(config)
dataset = SemanticKITTI(args=args)
dataloader = DataLoader(dataset, 1, False)
cur_data = next(iter(dataloader))

# cur_data['velodyne_proj_img0'][cur_data['denser_coordinate'][:,1], cur_data['denser_coordinate'][:,0]] = 500.
# show_img = topil(cur_data['velodyne_geo_s1'][0])
# show_img.show()
# img_tensor = totensor(np.expand_dims(cur_data['velodyne_proj_img0'], -1))
# img_tensor.shape

model = DepthCompletionModel(args).cuda()
model = model.load_from_checkpoint('./best_2dpapenet.ckpt', args=args, strict=False).cuda()
output_data = model(cur_data)
refine_img = output_data['refined_depth0']
img = np.squeeze(refine_img.data.cpu().numpy())
img = (img * 256.).astype('uint16')
img_buffer = img.tobytes()
imgsave = Image.new("I", img.T.shape)
imgsave.frombytes(img_buffer, 'raw', "I;16")
imgsave.save('./saveimg.png')
