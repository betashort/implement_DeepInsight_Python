# gene-expression
Analyze gene-expression

Main thema is using DeepInsight for gene-expression.

## Dataset
Use Kaggle Dataset "gene-expression"
URL:https://www.kaggle.com/crawford/gene-expression

## DeepInsight
URL:https://www.nature.com/articles/s41598-019-47765-6

# Create Images from table Dataset

## Require Python Library
### Defalt Library
1. sys
2. math  

### Add  
1. numpy
2. pandas
3. matplotlib
4. scipy
5. PIL
6. sklearn

## How to Use
```python
import DeepInsight

#table Dataset
deepinsight = DeepInsight.DeepInsigh()
train_image = deepinsight.fit(train_df, method='kpca')
test_image = deepinsight.predict(test_df)
```
