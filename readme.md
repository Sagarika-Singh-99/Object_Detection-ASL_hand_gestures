## Fine tuning YOLO to detect and translate ASL

## Overview
Meta’s Seamless M4T AI model, released in August 2023, translates voice and text into 100 languages. Unfortunately, this model fails to incorporate American Sign Language (ASL). For a translation model to have improved accessibility, video has to be one of the modalities, since real-time translation of ASL to English will require detecting hand gestures. Training an object detection model to recognize ASL gestures is a crucial first step towards real-time translation, even though ASL translation would require further processing, such as forming sentences given prior context.

The objective for this project is to finetune a pre-trained object detection model (YOLO-NAS S) to collect performance benchmarks. This will be achieved by running the model on video and images to translate hand gestures in ASL to their label - the corresponding English translations. The fine-tuning and inference will be run on a single GPU environment using Google Colab.


## Dataset
Roboflow ASL dataset

  
## Methodology
1. Gather dataset by Roboflow API.
2. Split the dataset and set batch size for training data.
3. Use super_gradient API to get the pre-trained model.
4. Evaluate best model on testing data.
5. Fine-tine model on training parameters using super_gradients trainers.
6. Set training hyperparameter and loss metric


## Usage
To run the code, copy the Python file into a Jupyter Notebook (Google Colab was used to train the model)

To train the model from scratch, you may need to allow access to your drive, where the model weights are stored to resume training

You will also need a roboflow API key to download the dataset. 



## Contributors
SS
