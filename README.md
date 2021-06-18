# The idea of mango competition
![](https://i.imgur.com/r9tRPFl.png)

## FOR TRAINING:

### main_1batch.ipynb : 
Due to the lackness of RAM, its goal is loading raw data and preprocessing them into shape “2800, 3, 224, 224” for each batch with type1 data argumentation, and save them in ’dataset/trainX_batch1.pt’, ‘dataset/trainy_batch1.pt’...’’dataset/trainX_batch4.pt’, ‘dataset/trainy_batch4.pt’ respectively.

### main_2batch.ipynb :  
Due to the lackness of RAM, its goal is loading raw data and preprocessing them into shape “2800, 3, 224, 224” for each batch with type2 data argumentation.and save them in ’dataset/trainX_batch5.pt’, ‘dataset/trainy_batch5.pt’...’dataset/trainX_batch8.pt’, ‘dataset/trainy_batch8.pt’ respectively.

### stacking1_model1.ipynb : 
Due to the lackness of RAM, I cannot import all data at a time, so I split it into 2 parts, the first part is for the outcome in main_1batch.ipynb.After importing all data, its time to construct model1, save the model parameters and optimizer into “model_param/resnet_cv_dev1”,  and the same as other files named “stacking1_model2~4”.

### stacking1_model5.ipynb : 
Due to the lackness of RAM, I cannot import all data at a time, so I split it into 2 parts, the second part is for the outcome in main_2batch.ipynb.After importing all data, its time to construct model5,  save the model parameters and optimizer into “model_param/vgg_cv_dev5”,  and the same as other files named “stacking1_model5~8”.

### stacking1_save_trainA_metadata.ipynb : 
we can consider the new batches from main_1batch.ipynb as A, just for easier to note. Now, in this file, we will construct the metadata with those models we trained just. use model1~4 to do feature extraction in dataA, and stacking all the features, we will obtain metadatas named “metadata1”, labels are similar as above, named “metalabels1”.

### stacking1_save_trainB_metadata.ipynb : 
we can consider the new batches from main_2batch.ipynb as B, just for easier to note. Now, in this file, we will construct the metadata with those models we trained .  We use model5~8 to do feature extraction in dataB, and stacking all the features, we will obtain metadatas named “metadata2”, labels are similar as above, named “metalabels2”.

### stacking2_model1.ipynb : 
this file imports “metadata1” and “metalabels1” and training the model  with logisticregression, and save the model and outcome  into “model_params/stacking2_logistic1” and “dataset/metaX_train1_stacking2” respectively. Labels are the same as above.

### stacking2_model2.ipynb : 
this file imports “metadata2” and “metalabels2” and training the model  with logisticregression, and save the model and outcome  into “model_paramsmodel_params/stacking2_logistic2” and “dataset/metaX_train2_stacking2” respectively. Labels are the same as above.

### stacking3_model.ipynb : 
we can load the data “metaX_train1_stacking” and “metaX_train2_stacking2”, and stack them vertically, now we have a matrix with shape “22400, 3”, training a model with lightGBM(“model_params/lgbm.pkl”), and testing with dev set, get 77.625% accuracy, and the dev set will demonstrate below.


## FOR TESTING:

### stacking1_save_dev_metadata.ipynb：
load all model we trained, and put all data into model1 to model4, named “dataset/metaX_dev1" and  initial labels “dataset/metay_dev1"; put all data into model5 to model8, named “dataset/metaX_dev2" and  initial labels “dataset/metay_dev2".

### stacking2_model1.ipynb : 
Do you remember the logisticregression model above in A? Here,“dataset/metadata1" and  “dataset/metalabel1” are training data here, and testing data is “dataset/metaX_dev1" and  “dataset/metay_dev1” , and the outputs are in “metaX_train1_stacking2”, “metaX_test1_stacking2” respectively, and the same as labels.

### stacking3_model.ipynb : 
For testing, we will load the dataset "dataset/metaX_test1_stacking2” and "dataset/metaX_test2_stacking2” from partA and partB respectively, here we won’t stack them just like training data above, we add they both, and divide by 2 because they both are the same datas in fact.


