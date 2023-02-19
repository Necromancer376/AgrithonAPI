import pandas as pd
import numpy as np

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import disease_map, details_map
import scipy.integrate as integrate


model = load_model('model.h5')

def predict(test_dir):
    test_img = [f for f in os.listdir(os.path.join(test_dir)) if not f.startswith(".")]
    print(test_img)
    # test_img = [test_dir]
    test_df = pd.DataFrame({'Image': test_img})
    print(test_df)
    
    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        test_dir,
        x_col = 'Image',
        y_col = None,
        class_mode = None,
        target_size = (256, 256),
        batch_size = 20,
        shuffle = False
    )
    predict = model.predict(test_generator, steps = np.ceil(test_generator.samples/20))
    # predict = model.predict(test_dir)
    test_df['Label'] = np.argmax(predict, axis = -1) # axis = -1 --> To compute the max element index within list of lists
    test_df['Label'] = test_df['Label'].replace(disease_map)

    prediction_dict = {}
    for value in test_df.to_dict('index').values():
        image_name = value['Image']
        image_prediction = value['Label']
        prediction_dict[image_name] = {}
        prediction_dict[image_name]['prediction'] = image_prediction
        prediction_dict[image_name]['description'] = details_map[image_prediction][0]
        prediction_dict[image_name]['symptoms'] = details_map[image_prediction][1]
        prediction_dict[image_name]['source'] = details_map[image_prediction][2]
    return prediction_dict


print(predict("./test_data"))
# print(predict("Apple_scab.jpg"))






# from io import BytesIO

# import torch
# from torchvision import datasets, transforms, models  # datsets  , transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import torchvision.transforms.functional as TF

# class CNN(nn.Module):
#     def __init__(self, K):
#         super(CNN, self).__init__()
#         self.conv_layers = nn.Sequential(
#             # conv1
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),
#             # conv2
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),
#             # conv3
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2),
#             # conv4
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2),
#         )

#         self.dense_layers = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(50176, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(1024, K),
#         )

#     def forward(self, X):
#         out = self.conv_layers(X)
#         out = out.view(-1, 50176)
#         out = self.dense_layers(out)
#         return out

# targets_size = 39
# model = CNN(targets_size)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))

# def single_prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     print(index)

# single_prediction("Apple_scab.jpg")

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image