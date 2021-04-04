# Classification-Computational-Intelligence
The purpose of this assignment is to investigate the ability of TSK (Takagi-Sugeno-Tang) models to solve classification problems. Specifically, two datasets are selected from the [UCI Repository](https://archive.ics.uci.edu/ml/index.php), in order to classify the available samples in their respective class using fuzzy neural models.

## Part 1
The first dataset is used for a simple investigation of the training and evaluation process of such models, as well as illustrating manners of analyzing and interpreting the results. We examine four TSK models in which the parameter that defines the number of the fuzzy IF-THEN rules, namely, the cluster radius, is modified. For the input partition we use the **Subtractive Clustering (SC)** method. Especially, in the first case (first two models) the SC will run for the whole training set (**class independent**), whereas in the second case (the remaining two models) the clustering will be applied to the training data for every class individually (**class dependent**). The reason why we use this approach of clustering is the increase of the model's interpretability and the production of "cleaner" clusters (hence the rules).  

###### dataset: [Haberman's Survival dataset](https://archive.ics.uci.edu/ml/datasets/haberman's+survival)

## Part 2
The second and more complicated dataset is used for a more complete modelling process, which involves, among others, **preprocessing steps** such as **feature selection** and methods for optimizing models through **cross validation**. Due to the large size of the dataset, problems such as rule explosion may appear. In order to avoid that, it is necessary that we decrease its dimensionality by choosing the most significant features and reject the less useful ones. After that, we apply **Grid Search** and **5-fold Cross Validation** to find the best combination of the number of features and cluster radius, which leads to the minimum validation error. In this part the SC is applied to the training data for every class individually (**class dependent**), as it seems to be a more efficient approach. Using the results that arise from the aforementioned search, we train the final model and we evaluate it according to the **error matrix** and the **accuracy** metric.

###### dataset: [Epileptic Seizure Recognition dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

## Code
Τhe files have been created using the **MATLAB R2018a** version. If a different version is used, you may need to customize some commands.

## Contact
If there are any questions, feel free to [contact me](mailto:thomi199822@gmail.com?subject=[GitHub]%20Source%20Han%20Sans). 
