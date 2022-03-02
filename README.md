The kNN takes two forms regression or classification.

Neighbourhood is cluster of items in n dimensions.

We can choose K by
Guessing
Heuristics for picking K
Use coprime class and K combinations
Pick coprime number of classes will ensure few ties.
Coprime numbers are two numbers that don’t share any common divisors except 1
Pick a K of at least 3
Choose a K that is greater or equal to number of classes plus one
Choosing a K that is low enough to avoid noise
If K is equal to number of classes then you would select most common class

There are many algo showing how to optimize K over a training set. They are ranging from genetic algo to brute force to grid searches.
Many suggest that you should determine K based on domain knowledge. 
The approach in which you are trying to minimize error based on arbitrary K is known as hill climbing problem. The idea is to iterate through a couple of possible Ks until you find suitable error. 
The difficult part about finding a K using an approach like genetic algo or brute force is that K increases and complexity of classification also increases and slows down performance. 

To construct KNN regression we will utilize something called KDTree. The idea is that KDTree will store data in a easily queriable fashion based on distance. We will use Euclidean distance because its easy to compute and will suit fine. 

Refer the code file.
