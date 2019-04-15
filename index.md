---
layout: home
title: Home
---

## Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age.
This website demonstrates the project made by Rohit Sarkar for the course
Digital Medicine taught by Professor Raghupathi Cavale

### Contents : 
- [Home]({{site.base_url}}) : Summary of the Project
- [Predict]({{ "/predict" | relative_url}}) : Input the features and get predictions in real time!
- [Report]({{ "/report" | relative_url}}) : A technical report of the Data Analysis and Machine Learning  process in the form of a `ipython` notebook
- [Synopsis]({{ "/synopsis" | relative_url}}) : Synopsis of the Project

### Brief
Worldwide, breast cancer is the second most common type of cancer after lung cancer and the fifth most common cause of cancer
death Breast Cancer is the leading type of cancer in women globally accounting for 25% of all cases.
Mammography is the most popular method for breast cancer screening available today.
However the results of breast biopsies resulting from the Mammogram Interpretations leads to approximately
***70%*** benign outcomes which could have been prevented.

The goal of this Project is to provide a Computer Aided Diagnosis (CAD) method that helps physicians in classifying
Mammographic Masses with the help of [BI-RADS](https://radiopaedia.org/articles/breast-imaging-reporting-and-data-system-bi-rads)
attributes and jthe Age of the patient. The machine learning model presented in this Project achieves a false positive rate of 
***15%*** using a Artificial Neural Network model which is significantly better than the physicians performance.

### Methodology
The dataset used for learning is the [Mammographic Mass Dataset](http://archive.ics.uci.edu/ml/datasets/mammographic+mass) from the UCI Machine Learning Repository. 
Data exploration and analysis was carried out to generate useful insights.

For the purpose of prediction an optimal classifier (Artificial Neural Network) was selected among 3 classifers.

The final model was deployed on a server running Python. A RESTful API was constructed using Python and the 
[Flask](http://flask.pocoo.org/) microframework. The frontend ie this website was made using HTML, CSS, JS, Jekyll and Github Pages.
For prediction the FrontEnd sends a request containing a list of features to the API Endpoint. The API then returns the 
predictions back to the Frontend.

### Code
The code used to analyse the data and build the Neural Network model can be found in the [report]({{ "/report" | relative_url}})
<br>
The code used to make this website and the webserver running the model can be found [here](https://github.com/rsarky/Mammographic-Mass/)

### Conclusion and Future Goals
It is clear that Computer Aided Diagnosis methods can significantly aid physicians in taking informed decisions. 
Thus Digital Medicine is playing a crucial role in the field of Breast Cancer.
In the future with more data, better models can be trained which will lead to higher accuracy.

### Open Source Libraries USed
- Scikit Learn:  Machine learning and Data analysis
- Seaborn: Probabilistic graphs
- Pandas: Data storage and manipulation
- Numpy: Numerical computing
- Matplotlib: Plotting
- Flask: Microframework for web server

### Acknowledgements
I would like to thank Professor Raghupathi Cavale and Professor Fayaz Shradgan for their guidance throughout the Digital Medicine Course.
