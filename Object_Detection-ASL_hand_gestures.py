#Google colab
from google.colab import drive
drive.mount('/content/drive', force_remount=True)


from roboflow import Roboflow
from tqdm.auto import tqdm
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.metrics import (
    DetectionMetrics_050,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import Trainer, models

# train model if true, inference otherwise
training_mode = False

# save to google drive to resume training
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoint3'
checkpoint_path = "/content/drive/MyDrive/checkpoint3/asl_run_3/RUN_20231203_025452_320762/ckpt_best.pth"

# save dataset to drive to avoid multiple downloads
API_KEY = "APIKEY"
def download_dataset():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(
        "majorproject-25tao").project("american-sign-language-v36cz")
    dataset = project.version(2).download("yolov8")
    return dataset

dataset = download_dataset()

# dataset path is /content for colab
dataset_path = '/content/American-sign-language-2/'
dataset_params = {
  'data_dir': dataset_path,
  'train_images_dir':'train/images',
  'train_labels_dir':'train/labels',
  'val_images_dir':'valid/images',
  'val_labels_dir':'valid/labels',
  'test_images_dir':'test/images',
  'test_labels_dir':'test/labels',
  'classes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'additional', 'alcohol', 'allergy', 'bacon', 'bag', 'barbecue', 'bill', 'biscuit', 'bitter', 'bread', 'burger', 'bye', 'cake', 'cash', 'cheese', 'chicken', 'coke', 'cold', 'cost', 'coupon', 'credit card', 'cup', 'dessert', 'drink', 'drive', 'eat', 'eggs', 'enjoy', 'fork', 'french fries', 'fresh', 'hello', 'hot', 'icecream', 'ingredients', 'juicy', 'ketchup', 'lactose', 'lettuce', 'lid', 'manager', 'menu', 'milk', 'mustard', 'napkin', 'no', 'order', 'pepper', 'pickle', 'pizza', 'please', 'ready', 'receipt', 'refill', 'repeat', 'safe', 'salt', 'sandwich', 'sauce', 'small', 'soda', 'sorry', 'spicy', 'spoon', 'straw', 'sugar', 'sweet', 'thank you', 'tissues', 'tomato', 'total', 'urgent', 'vegetables', 'wait', 'warm', 'water', 'what', 'would', 'yoghurt', 'your']
}

train_data = coco_detection_yolo_format_train(
  dataset_params={
      'data_dir': dataset_params['data_dir'],
      'images_dir': dataset_params['train_images_dir'],
      'labels_dir': dataset_params['train_labels_dir'],
      'classes': dataset_params['classes']
  },
  dataloader_params={
      'batch_size':16,
      'num_workers':2
  }
)

val_data = coco_detection_yolo_format_val(
  dataset_params={
      'data_dir': dataset_params['data_dir'],
      'images_dir': dataset_params['val_images_dir'],
      'labels_dir': dataset_params['val_labels_dir'],
      'classes': dataset_params['classes']
  },
  dataloader_params={
      'batch_size':16,
      'num_workers':2
  }
)

test_data = coco_detection_yolo_format_val(
  dataset_params={
      'data_dir': dataset_params['data_dir'],
      'images_dir': dataset_params['test_images_dir'],
      'labels_dir': dataset_params['test_labels_dir'],
      'classes': dataset_params['classes']
  },
  dataloader_params={
      'batch_size':16,
      'num_workers':2
  }
)

def plot_dataset():
    train_data.dataset.plot()

model = models.get('yolo_nas_s',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )

hyperparameters = {
    'silent_mode': True,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 35,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50',
    "resume": True
}


trainer = Trainer(experiment_name='asl_run_3', ckpt_root_dir=CHECKPOINT_DIR)

if (training_mode):
    trainer.train(model=model,
                  training_params=hyperparameters,
                  train_loader=train_data,
                  valid_loader=val_data)

elif (checkpoint_path != ""):
    best_model = models.get('yolo_nas_s',
                            num_classes=len(dataset_params['classes']),
                            checkpoint_path=checkpoint_path)

trainer.test(model=best_model,
             test_loader=test_data,
             test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                    top_k_predictions=300,
                                                    num_cls=len(dataset_params['classes']),
                                                    normalize_targets=True,
                                                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                               nms_top_k=1000,
                                                                                                               max_predictions=300,
                                                                                                               nms_threshold=0.7)))

# predict images and videos, save to output_path
def predict(model, input_path, output_path):
    model.predict(input_path).save(output_path)
