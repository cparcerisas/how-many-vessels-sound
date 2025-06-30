try:
    import comet_ml
except ModuleNotFoundError:
    comet_ml = None
import yaml
import torch
from ultralytics import YOLO
import os


def run():
    YAML_FILE = './yolo_config.yaml'
    project_name = 'irene_vessels' # Change to the name of your run

    # Check if CUDA is available
    print('CUDA device count:')
    print(torch.cuda.device_count())

    # Read the config file
    with open(YAML_FILE, 'r') as file:
        config = yaml.safe_load(file)

    if "COMET_API_KEY" in os.environ:
        comet_ml.login()
        experiment = comet_ml.start(api_key=os.environ['COMET_API_KEY'], project_name=project_name)
    else:
        print(os.environ)
        print('No comet_ml API found, not logging this round...')

    # Load a model
    model = YOLO('yolo11n.pt')

    # train the model
    best_params = {
        'iou': 0.3,
        'imgsz': 640,
        'hsv_s': 0,
        'hsv_v':  0,
        'degrees': 0,
        'translate': 0,
        'scale': 0,
        'shear': 0,
        'perspective': 0,
        'flipud': 0,
        'fliplr': 0,
        'bgr': 0,
        'mosaic': 0,
        'mixup': 0,
        'copy_paste': 0,
        'erasing': 0,
        'crop_fraction': 0,
    }
    model.train(epochs=50, batch=32, data=YAML_FILE,
                project=project_name, save_dir=config['path'] + '/runs/detect/' + project_name, resume=False, **best_params)

    if "COMET_API_KEY" in os.environ:
        experiment.end()


if __name__ == '__main__':
    run()