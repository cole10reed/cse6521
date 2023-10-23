# cse6521
AI Project for CSE6521

Our project has 3 main tasks:

Task 1:  We will apply three pre-trained Segment Anything models (ViT-H, ViT-L, and ViT-B) to each of the datasets and evaluate their performance.  One compatibility issue with the building extraction problem lies in its name.  The model will segment every distinct object that it detects in an image, regardless of whether it is a building.  To combat this, our plan is to measure the performance only on the correctness of the building-specific segments. 

 

Task 2: We will take the best performing SA model from Task 1 and fine tune the model to our building segmentation datasets.  The purpose of the fine tuning is to obtain a slightly higher performance on data which the model has not seen before.  We will then compare the fine-tuned model’s performance with that of the original model. 

 

Task 3:  Due to the SA model’s inability to classify the segments that it generates, we will create a convolutional neural network and train it to classify the segments of the SA model.  For this task we will use the SA model that shows the best performance in Tasks 1 and 2 (ViT-H, ViT-L
, or ViT-B). 


Task 1 can be found in TaskOne.py. Each of the 3 models (ViT-H, L, and B) are applied to all the datasets and the results are aggregated on results.py

