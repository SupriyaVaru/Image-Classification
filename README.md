# Image Classification using AWS SageMaker

Used AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. I have used resnet50 pretrained model for the dog breed classication data set. 

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 
train_and_deploy_1 is the main python notebook for this project.
hpo(1).py is used as entrypoint for hyperparameter tuning job.This doesn't have debugging and profiling hooks.
train_model_1.py is used as entry point to train the model using best hyperparametersis and debugging, profiling hooks are introduced in the code.
inference.py is used to deploy model to endpoint and draw inference.
At the end I have tried to predict two dog images and showed the results.

## Dataset
The provided dataset is the dogbreed classification dataset.
It has train/test/valid image folders consisting of 133 dog breeds images subfolders.
Each subfolder has mutliple dog images of a breed.

### Access
Uploaded the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

ResNet50 is a popular pre-trained model for image classification because it has demonstrated state-of-the-art performance on large-scale image classification tasks such as the ImageNet dataset. ResNet50 is a deep neural network architecture that consists of 50 layers, and it is a variant of the ResNet family of models, which are designed to solve the problem of vanishing gradients during training of deep neural networks.

I have choosed ResNet50 pretrained model to apply transfer learning because pre-trained ResNet50 model is capable of extracting high-level features from images, which can be fine-tuned for our specific image classification task. This makes ResNet50 a good choice for transfer learning, where we can leverage the pre-trained weights of the model to achieve good performance on our image classification task with a relatively small amount of training data.




Remember that your README should:
- Include a screenshot of completed training jobs
-
![image](https://user-images.githubusercontent.com/103468158/220745575-d204c103-438d-413a-a1df-7ea02dbb9220.png)

![image](https://user-images.githubusercontent.com/103468158/220746049-d6c7d85a-d8e2-4044-b58f-d85ff51108fe.png)

- Logs metrics during the training process
Log metrics of 4 Hyperparameter tuning jobs
![image](https://user-images.githubusercontent.com/103468158/220747837-2d150dc2-2d65-4ad7-a9db-90025d992a03.png)
![image](https://user-images.githubusercontent.com/103468158/220748206-92b2de9a-cf3b-4ffe-94ae-53d5322b079e.png)
![image](https://user-images.githubusercontent.com/103468158/220748450-955c88c3-05b1-424b-94be-2926202e6876.png)
![image](https://user-images.githubusercontent.com/103468158/220748748-dfc038e8-30a9-4299-b857-51fd0a58e37c.png)

- Tune at least two hyperparameters

Tuned three Hyperparameters
Below are the hyperparameters and their ranges that I have choosen to search

"learning_rate": (0.001, 0.1)
"batch_size": (32, 64, 128, 256, 512)
"num_epochs": (2, 4)

- Retrieve the best best hyperparameters from all your training jobs
-
![image](https://user-images.githubusercontent.com/103468158/220746202-72e9b733-6805-4c84-a078-800cdeb4d5b1.png)

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
