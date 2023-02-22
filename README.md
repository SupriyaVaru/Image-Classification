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

  ![image](https://user-images.githubusercontent.com/103468158/220750022-e63bcdd2-b4e9-4c06-a25c-083d3efadbe7.png)

  ![image](https://user-images.githubusercontent.com/103468158/220746202-72e9b733-6805-4c84-a078-800cdeb4d5b1.png)

## Debugging and Profiling
   Give an overview of how you performed model debugging and profiling in Sagemaker
   I choosed to monitor overfitting, overtraining, poor_weight initilization, loss_not_decreasing, LowGPUutilization and profiler report which i have attached in          files using debugging and profiling.
   First I have setup debugging and profiling rules and hooks.
   Included these parameters in estimator of training job.
   Also in train_model_1.py script imported SMDebug framework class, pass the hook to train and test function, set the hook for training and evaluation phase.
   Then I have registered hooks to save output tensors.

### Results
   What are the results/insights did you get by profiling/debugging your model?
   
   Debugging results:
   
   ![image](https://user-images.githubusercontent.com/103468158/220751666-ecce4050-af56-446a-8d81-b186f0207a55.png)
 
   1)"PoorWeightInitialization: Error" has been observed in logs. This can be fixed by using weight intialization techniques like Xavier/Glorot Initialization, He          Initialization, Uniform Initialization etc.,           
   2)From the output it can be seen that training loss decreases with increase in steps whereas validation loss is almost constant and very low compared with training      loss. This could be sign of overfitting. We can use regularization techniques to avoid this.
   
   Profiling results:
   ![image](https://user-images.githubusercontent.com/103468158/220752249-6429391f-6e8e-4d18-a2bc-264d46f4f491.png)
   
   ![image](https://user-images.githubusercontent.com/103468158/220752641-d5337cca-12a1-4294-b4e2-9954d248bcef.png)


Remember to provide the profiler html/pdf file in your submission-attached


## Model Deployment
   Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
    
  -The deployed model for dog breed classification is based on the ResNet50 architecture, which is a popular convolutional neural network (CNN) model for image           classification tasks.
  -To improve its performance on the specific task of dog breed classification, the ResNet50 model was pretrained on the large-scale ImageNet dataset, which contains     millions of labeled images across a wide range of categories.
  -The pretrained ResNet50 model was then fine-tuned on the dog breed classification dataset, which consists of over 16,000 labeled images of dogs belonging to 133       different breeds.
  -The input to the model is a 3-channel image of size 224x224 pixels, which is a common size used in many image classification tasks.
  -The output of the model is a vector of 133 values, each representing the probability that the input image belongs to one of the 133 dog breeds in the dataset.
  -The model does not apply softmax or log softmax to its output, as these operations are only applied during the calculation of the loss function during training.
  -To obtain the predicted dog breed for a given input image, the index of the maximum value in the output vector is used as the predicted label.
  -The model was fine-tuned for one epoch using 'learning_rate': 0.0020610662962159635, 'batch_size': 256, 'num_epochs': 4.
  -We have got Test set: Average loss: 0.0027, Accuracy: 694/836 (83%)
  
  Training job log metrics:
  ![image](https://user-images.githubusercontent.com/103468158/220755931-58376ef3-22a9-4eb2-82de-0e8572870a4c.png)

  ![image](https://user-images.githubusercontent.com/103468158/220755519-9a9eadc9-be69-44ab-84cf-a3289b9e5511.png)
  ![image](https://user-images.githubusercontent.com/103468158/220755802-cf66d15f-a3fe-4b55-9440-4cd96e32514a.png)

  Used inference.py and to deploy model to end point.
  ![endpoint-inference](https://user-images.githubusercontent.com/103468158/220756437-44b89335-6698-469d-8322-a74c7f5d3840.png)

  Preprocessed image and used predictor to predict the dog breed.
  ![image](https://user-images.githubusercontent.com/103468158/220756561-5d0813cd-0e0d-4a60-90e4-d3b9902700f2.png)
  ![image](https://user-images.githubusercontent.com/103468158/220756659-575fd6af-9adf-49c7-93e4-574ce688cdcb.png)

  
## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
