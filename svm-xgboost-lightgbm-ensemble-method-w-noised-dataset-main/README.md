# SVM, XGB, LGBM, Ensemble Method w/Noised-Dataset
I tried to deal with the overfitting problem using only the AI-generated training set. 

<br>

There are several models used in this dataset that are overfitting.
Generally, 100% accuracy is not sensible in Machine Learning. There might be many reasons for this problem. 
It is a fact that using only generated images for training is not good in terms of model accuracy due to the overfitting problem.
I used deep learning models, linear kernal SVM, XGBoost, Radial basis function kernel SVM, LightBoost with best parameters stemmed from hyperparameter tuning, Voting classifier with another classifiers.
<br>
 
 <mark>
I tried 276 principal components and got sufficient results. However, in 2 principal components, results were very good too. This is very compressed data. Maybe this situation is proof of the not-so-good side of the dataset.</mark><br>
<br>
I investigated some papers and posts. I agree with this comment by Zac Yauney below this article (https://machinelearningmastery.com/mostly-generate-synethetic-data-machine-learning-why/).

Part of this comment;
> ... The reason I expect synthetic data to become less useful for training better models is that the goal of a machine learning method is to learn the shape of an underlying statistical distribution, and synthetic data only reflects the algorithm’s current hypothesis about that distribution given the real data it was trained on. ...



## ❗My idea
<p>I tried to <b> increase the entropy of images </b> in datasets by adding random noise. Simply put, entropy is a measurement of uncertainty. I wanted to spoil the data with some uncertainty because it may help our model to generalize and yield a sensible accuracy score without overfitting. </p>

![band_daniel_noised_images_graph](https://github.com/john-fante/SVM-XGB-LGBM-Ensemble-Method-w-Noised-Dataset/assets/50263592/fd40fcbb-d4e0-4adb-b00c-c0b14b09a702)


<br>
<span style="font-size:1em; color:gray;"> <i> Figure 1: An original image and its noised samples  </i> </span>

<hr>

## Results 

| <b>Model</b>      | <b>Test Acc <span style='color:#e74c3c'>(2 Comp. PCA)</span></b> | <b>Test Acc <span style='color:#e74c3c'>(276 Comp. PCA)</span></b> | 
| :---------------- | :----------- |:----------- |
| Linear SVM        |   96.75 %   |  95.25 %   | 
| LigthGBM (Tuned)  |   93 %   | 95.5 %| 
| RBF SVM           |     96 %   |  92.5 %  | 
| XGBoost           |     87.25 %   |  95.5 %  | 
| Voting Classifier       |     94.25 %   |  95.5 % | 
