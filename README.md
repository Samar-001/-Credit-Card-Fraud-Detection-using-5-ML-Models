# SAMPLING: Credit Card Fraud Detection
## Using 4 sampling methods to predict the result with the help of 5 ML models
### _Sampling Methods used: Random, Systematic, Stratified, Clustered_
### _ML Models used: Logistic Regression, Decision Trees, Random Forest, Naive Bayes, K-Nearest Neighbor_
Done By: **Samarjot Singh  102003242**

***


## Checking for imbalanced data

**TOPSIS**, known as Technique for Order of Preference by Similarity to Ideal Solution, is a multi-criteria decision analysis method. It compares a set of alternatives based on a pre-specified criterion. The method is used in the business across various industries, every time we need to make an analytical decision based on collected data. More details at [YouTube](https://www.youtube.com/watch?v=kfcN7MuYVeI&ab_channel=ManojMathew).

<br>

## How to run this package:

TOPSIS-Samar 102003242  can be used by running following command in CMD:

```
>> topsis 102003242-data.csv "1,1,1,2,1" "-,+,+,-,+" 102003242-result.csv
```

<br>

## Sample dataset

The decision matrix should be constructed with each row representing a Fund Name, and each column representing a criterion P1, P2, P3, P4, P5.

Fund Name | P1 | P2 | P3 | P4 | P5
------------ | ------------- | ------------ | ------------- | ------------- | ------------
M1 |	0.72 | 0.52	| 4.4 | 62.1 | 16.94
M3 |	0.72 | 0.52	| 5.7 | 48.6 | 13.91
M2 |	0.76 | 0.58	| 4.2 | 39.4 | 11.21
M4 |	0.68 | 0.46	| 6.7 | 50 | 14.46
M5 |	0.67 | 0.45	| 5.2 | 62.2 | 17.13
M6 |	0.86 | 0.74	| 5.2 | 63.8 | 17.65
M7 |	0.93 | 0.86	| 4.5 | 65.6 | 17.97
M8 |	0.78 | 0.61	| 5.4 | 69.7 | 19.12

Weights(`w`) and Impacts(`i`) will be applied later in the code.

<br>

## Output

```
 Row No.   Performance Score    Rank
--------  -------------------  ------
  3            0.332629          8
  2            0.555383          1
  1            0.548848          2
  4            0.530816          3
  5            0.354290          6
  6            0.421567          5
  7            0.435080          4
  8            0.353907          7
```
<br>
The rankings are displayed in the form of a table with the 1st rank offering us the best decision and last rank offering the worst decision making, according to TOPSIS method.
