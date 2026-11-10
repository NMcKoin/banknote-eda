import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load and preview the dataset
banknotes = pd.read_csv('./data/banknote-auth.csv')

print('\nDataset Shape:')
print(banknotes.shape)

print('\nDataset information:')
print(banknotes.info())

print('\nFirst 10 rows of the dataset:')
print(banknotes.head(10))

print('\nMissing Values:')
print(banknotes.isnull().sum())

# Map the class labels to more descriptive names
banknotes['Authenticity'] = banknotes['label'].map({
  0: 'Authentic',
  1: 'Counterfeit'
  })


# Class Distribution
class_counts = banknotes['Authenticity'].value_counts()

print('\nClass Counts:')
print(class_counts)

plt.figure(figsize=(8, 6))
sns.countplot(x='Authenticity',
              data=banknotes)
plt.title('Class Distribution of Banknotes')
plt.xlabel('Authenticity')
plt.ylabel('Count')
plt.savefig('./output/class_distribution.png')
plt.close()

# Scatter plot of variance vs skewness
plt.figure(figsize=(8, 6))
sns.scatterplot(x='variance',
                y='skewness',
                hue='Authenticity',
                data=banknotes)
plt.title('Scatter Plot of Variance vs Skewness')
plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.savefig('./output/scatter_plot_variance_skewness.png')
plt.close()

# Scatter plot of entropy vs kurtosis
plt.figure(figsize=(8, 6))
sns.scatterplot(x='entropy',
                y='kurtosis',
                hue='Authenticity',
                data=banknotes)
plt.title('Scatter Plot of Entropy vs Kurtosis')
plt.xlabel('Entropy')
plt.ylabel('Kurtosis')
plt.savefig('./output/scatter_plot_entropy_kurtosis.png')
plt.close()

# Correlation Matrix
features_df = banknotes.drop(columns=['Authenticity'])
correlation_matrix = features_df.corr()

print('\nCorrelation Matrix:')
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            square=True)
plt.title('Correlation Matrix of Features')
plt.savefig('./output/correlation_matrix.png')  
plt.close()

# The strongest correlation between input features is between skewness and kurtosis,
# with a value of -0.79 - a strong negative correlation. 
# As skewness increases, kurtosis tends to decrease, and vice versa.
# Most other features have weak or moderate correlations suggesting that the freatures provide distinct information.


# How balanced are the two classes?
#
#The dataset is pretty balanced, there are slightly more authentic banknotes compared to counterfeit ones.
#but the difference is not large enough to create a significant class imbalance issue.

# Which feature pair shows the best seperation?
#
# The variance and skewness feature pair shows the best separation between the two classes.
# The classes form distinct clusters with a moderate amount of overlap compared to the 
# entropy and kurtosis feature pair.

#Does the data appear linearly separable?
#
# The data mostly appears to be linearly separable, Some overlap does exist between
# the two classes. A linear decision boundary could be drawn to separate the two classes, but it may not perfectly classify all instances.

# How do you think a MLP will do against a Perceptron on this data?
#
# A perceptron should perform decently on this dataset. Because the classes show a large amount of linear separability.
# However, an MLP is expected to perform better than a perceptron, because it can learn more complex
# non linear relationships and handle overlapping regions more effectively.