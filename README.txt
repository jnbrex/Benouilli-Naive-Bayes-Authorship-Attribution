To compile: g++ -O3 -std=c++11 -o AAAC main.cpp

To run on the documents in the /problemA/ folder: ./AAAC AAAC_problems/problemA/
To run on the documents in a different folder, enter that directory in the command line instead

This program is a Bernoulli Naive Bayes Authorship Attribution program.  It uses a set of training data (passages labeled with an author ID) to deduce authors of test data (passages without labels).  It achieves this by analyzing the frequency of use of the words in stopwords.txt.  The program outputs the accuracy on the test data, the confusion matrix of the test data, the feature ranking of the top 20 features (ranked by class-conditional entropy), and the test accuracy for using X amount of features, from 10 to 423, with each iteration using 10 more features than the previous.

