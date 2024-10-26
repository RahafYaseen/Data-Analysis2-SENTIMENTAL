

##  Overview

This project implements a spam classification model using Natural Language Processing (NLP) techniques and machine learning algorithms. The goal is to classify text messages as either "ham" (non-spam) or "spam".

## Libraries Used

- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **NLTK**: For natural language processing tasks.
- **Scikit-learn**: For machine learning algorithms and utilities.
- **WordCloud**: For visualizing word frequency.
- **Plotly**: For interactive visualizations.

## Dataset
You can download the dataset from this link ([spambase.csv file](https://archive.ics.uci.edu/dataset/94/spambase)).




- **v1**: Indicates whether a message is "ham" or "spam".
- **v2**: The text content of the message.

### Data Preparation

1. **Data Loading**: The dataset is loaded using Pandas.
2. **Preprocessing**:
   - Removed unnecessary columns.
   - Renamed columns for clarity.
   - Handled missing and duplicate values.
   - Transformed text data by:
     - Converting to lowercase.
     - Tokenizing into words.
     - Removing stopwords and punctuation.
     - Applying stemming.

### Data Visualization

- **Pie Chart**: Displays the distribution of spam vs. ham messages.
- **Word Cloud**: Visualizes the most common words in spam and ham messages.
- **Bar Plots**: Show the top words in spam and ham messages.

## Model Training

### Text Vectorization

Text data is vectorized using the **TF-IDF** (Term Frequency-Inverse Document Frequency) method to convert text into numerical format suitable for machine learning algorithms.

### Algorithms Used

1. **Gaussian Naive Bayes (GNB)**
2. **Multinomial Naive Bayes (MNB)**
3. **Bernoulli Naive Bayes (BNB)**

### Model Evaluation

- **Accuracy**: Measured using accuracy score.
- **Confusion Matrix**: Visualized to assess model performance.
- **Precision**: Calculated to evaluate the model's performance on spam detection.
- **ROC Curve**: Plotted to visualize the performance of the classifiers.

### Results

The project evaluates the performance of the three models and compares their ROC curves and AUC scores.

## How to Run the Project

1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud plotly
   ```

2. Download the `spam.csv` dataset and place it in the same directory as the script.

3. Run the script:
   ```bash
   python spam_classification.py
   ```

## Conclusion

This project demonstrates a complete workflow for spam classification using machine learning. The steps from data preprocessing to model evaluation are essential for building a robust spam classifier.

## Acknowledgments

- Thanks to the contributors of the libraries used in this project.
- The dataset is a public resource and can be found on various data repository websites.

---

Feel free to modify or extend this README to better suit your project's needs!
