# Gamma/Hadron Separation w/XGBoost, LightGBM, SVM

<b> Project Goal : The main goal of the project is to distinguish gamma-ray events from hadronic background events in order to identify and study celestial gamma-ray sources accurately.

kaggle -> https://www.kaggle.com/code/banddaniel/gamma-hadron-separation-w-xgb-lgbm-svm
</b>



## ❗My idea
<p>Firstly, I split the most valuable feature (fAlpha) from all features in respect of Mutual Information Scores and Point-BiSerial Coefficients. Then, I applied PCA to remnant features and combined these principal components with the most valuable feature (fAlpha).<br>
As a result, this operation slightly improved the AUC ROC score.<br>
</p>

* <p><b>Mutual Information Scores</b> give information pertaining to non-linear relationships in features.</p>
* <p><b>Point-Biserial Correlation Coefficients</b> measure the association between a continuous variable and a binary variable.</p>
<br>

## Result

I used BayesSearchCV for tuning hyperparameters.

| <b>Model</b>      | <b>ROC AUC Score <span style='color:#e74c3c'></span></b> | 
| :---------------- | :----------- |
| XGBoost (Tuned)    |  0.88986    | 
| <mark>LightGBM</mark>   |   0.89312  | 
| RBF SVM   (Tuned)   |   0.88838  |

<br>

![257043772-88335814-e32c-4abb-a7ae-96967eb2263c](https://github.com/john-fante/my-machine-learning-projects/assets/50263592/db8ea4ec-4b50-42b9-a467-c485cd65eeeb)


## References
1. https://en.wikipedia.org/wiki/High_Altitude_Water_Cherenkov_Experiment
2. https://www.statisticshowto.com/point-biserial-correlation/
3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html
4. [Use of Machine Learning for gamma/hadron separation with HAWC](https://arxiv.org/pdf/2108.00112.pdf)
