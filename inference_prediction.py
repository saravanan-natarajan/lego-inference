import datetime
import json
import logging
import time

# import boto3
import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# # Change the path accordingly
# MODEL_DIR = "/models/"
# MODEL_FILE = "resnet50_dino_train_others.pth"
# CLASSES_FILE = "labels_dino_train_others.txt"

# MODEL_FILE = MODEL_DIR + MODEL_FILE
# CLASSES_FILE = MODEL_DIR + CLASSES_FILE
from helper_lib import args_parse, load_config, my_logger

args = args_parse()
if args.log:
    args.log = args.log.upper()
    print('log level set: {}'.format(args.log))

config_dict = load_config('./config/' + args.config)

log_file = config_dict['directories']['log'] + 'script' + '_' + datetime.utcnow().strftime('%Y%m') + '.log'
logger = my_logger(log_file, args.log)
logger.info('start')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# # Change the path accordingly
# MODEL_DIR = '/models/'
# MODEL_FILE = 'res50_dino_train.pth'
# CLASSES_FILE = 'res50_dino_train.txt'
# MODEL_FILE = MODEL_DIR + MODEL_FILE
# CLASSES_FILE = MODEL_DIR + CLASSES_FILE

MODEL_FILE = config_dict['directories']['model']
CLASSES_FILE = config_dict['directories']['classes']
logger.debug('MODEL_FILE: {}'.format(MODEL_FILE))
logger.debug('CLASSES_FILE: {}'.format(CLASSES_FILE))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

logger.debug('mean: {}'.format(mean))
logger.debug('std: {}'.format(std))

preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def load_model():
    logger.info('load_model: start')
    classes = open(f'{CLASSES_FILE}', 'r').read().splitlines()
    logger.info('classes loaded: {}'.format(classes))
    # model_path = f'{MODEL_FILE}'
    logger.info('model file: {}'.format(MODEL_FILE))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device used: {}'.format(device))
    # model = torch.jit.load(model_path, map_location=device)
    model = torch.load(MODEL_FILE, map_location=device)  # Modified by SN
    model = model.to(device)  # Modified by SN
    # To set dropout and batch normalization layers to evaluation mode before running inference.
    logger.info('model evel: {}'.format(model.eval()))
    logger.info('load_model: end')
    return model.eval(), classes


model, classes = load_model()


def predict(model, classes, image_tensor):
    """Predicts the class of an image_tensor."""

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Added by SN
    image_tensor = image_tensor.to(device)  # Added by SN
    predict_values = model(image_tensor)
    logger.info("Inference time: {} seconds".format(time.time() - start_time))
    softmaxed = F.softmax(predict_values, dim=1)
    probability_tensor, index = torch.max(softmaxed, dim=1)
    prediction = classes[index]
    probability = "{:1.2f}".format(probability_tensor.item())
    logger.info(f'Predicted class is {prediction} with a probability of {probability}')
    return {'class': prediction, 'probability': probability}


def image_to_tensor(img):
    # Transforms the posted image to a PyTorch Tensor
    logger.info("image_to_tensor: start")
    img = PIL.Image.open(img)  # Added by SN - In case if you are passing the Image
    img_tensor = preprocess_pipeline(img)
    img_tensor = img_tensor.unsqueeze(0)  # 3d to 4d for batch
    logger.info("image_to_tensor: start")
    return img_tensor


def inference(img):
    """The main inference function which gets passed an image to classify"""

    image_tensor = image_to_tensor(img)
    response = predict(model, classes, image_tensor)
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }


def main():
    print(inference('sample_images/dino_image1.png'))
    print(inference('sample_images/dino_image3.png'))
    print(inference('sample_images/others_image2.jpeg'))
    print(inference('sample_images/train_image4.png'))


if __name__ == '__main__':
    main()
