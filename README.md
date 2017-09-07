# record_linkage

#### Python scripts for unsupervised, pair-wise linkage of records from multiple databases.
These scripts make use of numerous probabilistic/ML algorithms commonly employed in record linkage. The output of field comparisons (using exact or 'fuzzy' matching) is fed into each network, for each possible pair of records between the 2 databases. During training, the target of each network is the actual relationship between 2 records (i.e. match/not match).

## - Data sources (updated monthly/semi-annually):
* **Patient discharge data** from participating hospital/state-agency.  
* **Birth certificate data** from state Vital Records department.  
Records are matched using both simple field comparisons, as well as complex comparisons that have several levels of matching, which allows the classifier to discriminate more accurately. 

### - Maximizing pairing scores (minimizing cost):
* Maximize pairing with **Kuhn-Munkres** (i.e. Hungarian) Algorithm ([more info])

## - Record-linkage algorithms:
* **Probabilistic record linkage**
Probablistic approach described by Jaro<sup>[1]</sup> for linking of large public health data files. 
These formulas are equivalent to the *naive Bayes classifier*, which depends upon the same independence assumptions.  
* **Single layer perceptron**
Machine learning approach described by Wilson<sup>[2]</sup> for linking of genealogical records.  
* **Multi-layer neural network**
Artificial neural network with backpropagation<sup>[3]</sup>.

1. Jaro, M. A. (1989). Advances in record-linkage methodology as applied to matching the 1985 census of Tampa, Florida. Journal of the American Statistical Association, 84(406), 414-420.
2. Wilson, D. R. (2011, July). Beyond probabilistic record linkage: Using neural networks and complex features to improve genealogical record linkage. In Neural Networks (IJCNN), The 2011 International Joint Conference on (pp. 9-14). IEEE.
3. (Various sources including:) Raschka, S. (2015). Python machine learning. Packt Publishing Ltd. Chicago	

[more info]: https://pypi.python.org/pypi/munkres/
