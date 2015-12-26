#include <glob.h>
#include <libgen.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <stack>
#include <utility>
#include "BernoulliNaiveBayes.h"
using namespace std;

//uses glob to collect of the files within a directory and put their names into a vector
inline vector<string> glob(const string& pat) {
	glob_t glob_result;
	glob(pat.c_str(), GLOB_TILDE, NULL, &glob_result);
	vector<string> ret;
	for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
		ret.push_back(string(glob_result.gl_pathv[i]));
	}
	
	globfree(&glob_result);
	return ret;
}

int main(int argc, char** argv) {
	assert(argc == 2);  //assert that the only input is the directory name
	string directoryName = argv[1];  //set string directoryName equal to the directory name
	PreProcess pp = PreProcess();  //initialize a PreProcess object
	
	//get letter
	string letter = directoryName.substr(directoryName.size() - 2, 1);
	
	//the vectors contain the names of all files within the given directory
	vector<string> trainingNames = glob(directoryName + letter + "train*");
	vector<string> sampleNames = glob(directoryName + letter + "sample*");
	
	//create the set of author IDs from the training examples
	unordered_set<int> authorIDs;
	int letterPos;
	for (int i = 0; i < trainingNames.size(); i++) {
		letterPos = trainingNames[i].find_last_of("n");
		authorIDs.emplace(stoi(trainingNames[i].substr(letterPos + 1, 2)));
	}
	
	//storing stop words in a vector	
	ifstream infile;
	infile.open("stopwords.txt");
	string word;
	vector<string> stopWords; //vector to store stop words
	while(getline(infile, word))
		stopWords.push_back(word);
	infile.close();

	//initialize a BNB object with the file names, stop words, author id set, and the PreProcess object
	BernoulliNaiveBayes BNB = BernoulliNaiveBayes(trainingNames, sampleNames, stopWords, authorIDs, pp, letterPos);
	
	//initialize the prior and condprob maps
	map <int, double> prior;
	map <string, map<int, double>> condprob;
	
	//call the training function
	BNB.train(prior, condprob);
	
	//create the list of predicted authors that corresponds one to one with the sample names list
	vector<int> predicted;
	for (int i = 0; i < sampleNames.size(); i++) {
		predicted.push_back(BNB.test(prior, condprob, sampleNames[i]));
	}
	
	//output the sample names and predicted authors
	for (int i = 0; i < sampleNames.size(); i++) {
		sampleNames[i] = sampleNames[i].substr(sampleNames[i].find_last_of("/") + 1);
		cout << sampleNames[i] << " " << predicted[i] << endl;
	}
	
	infile.open("test_ground_truth.txt");
	string findPhrase = letter + "sample";
	vector<string> authorList;
	while(getline(infile, word)) {
		if (word.find(findPhrase) != string::npos)
				authorList.push_back(word);
	}
	infile.close();
	//list of actual authors
	vector<int> actual;
	unordered_set<int> numAuthors;
	
	for (int i = 0; i < authorList.size(); i++) {
		actual.push_back(stoi(authorList[i].substr(authorList[i].length() - 2)));
		numAuthors.emplace(actual[i]);
	}
	
	//calculate the accuracy
	double numCorrect = 0;
	for (int i = 0; i < actual.size(); i++) {
		if (actual[i] == predicted[i])
			numCorrect++;
	}
	
	cout << endl << numAuthors.size() << " authors give " << numCorrect/double(actual.size()) * 100 << " % " << "accuracy.";
	
	//print the confusion matrix////////////////////////////////////////////////////////////////
	pp.printConfMat(actual, predicted);
	
	vector<double> featureRanking;
	//calculate CCE for each feature////////////////////////////////////////////////////////////
	int vocabSize = stopWords.size();
	for (int i = 0; i < vocabSize; i++) {						//iterate through each stop word
    	vector<int> Dc;
    	double CCE = 0;
		for (int j = 1; j < authorIDs.size() + 1; j++) {			//iterate through each author
			double Nc = 0;										//do Nc <- CountDocsInClass(D,c)
    		Dc.clear();
    		for (int k = 0; k < trainingNames.size(); k++) {
    			if (j == stoi(trainingNames[k].substr(letterPos + 1, 2))) {
    				Nc++;
    				Dc.push_back(k);
    			}			
    		}
    		double Pc = Nc/double(trainingNames.size());		//calculate Pc
    		
			double numDocsWithWord = 0;
    		for (int k = 0; k < Dc.size(); k++) {
    			ifstream infile;
    			infile.open(trainingNames[Dc[k]]);
    			string fileContent;
				//read entire file into string fileContent and and put it into fileWords vector
				fileContent.assign(istreambuf_iterator<char>(infile), istreambuf_iterator<char>());
				infile.close();
				fileContent = pp.remove_extra_space(fileContent);
				vector<string> fileWords = pp.word_tokenize(fileContent);
				
				//create an unordered set and fill it with file words
				unordered_set<string> fileWordSet;
				int fileWordsSize = fileWords.size();
				for (int l = 0; l < fileWordsSize; l++)
					fileWordSet.emplace(fileWords[l]);
				//check if word is in doc
    			if (fileWordSet.find(stopWords[i]) != fileWordSet.end())
    				numDocsWithWord++;
    		}
    		double Pfc = (numDocsWithWord + 1)/(Nc + 2);			//calculate Pfc
			CCE -= Pc * Pfc * log2(Pfc);
		}

		featureRanking.push_back(CCE);
	}
	
	vector<pair<double, string> > features;
	
	for (int i = 0; i < vocabSize; i++) {
		auto p1 = make_pair(featureRanking[i], stopWords[i]);
		features.push_back(p1);
	}
	
	//sort the vector from greatest to least
	sort(features.rbegin(), features.rend());
	
	cout << endl;
	for (int i = 0; i < 20; i++)
		cout << features[i].second << " " << features[i].first << "\n";
	
	//feature curve/////////////////////////////////////////////////////////////////////////////
	vector<vector<string> > allTrainingDocs;
	string fileContent;
	for (int i = 0; i < trainingNames.size(); i++) {
    	infile.open(trainingNames[i]);
		fileContent.clear();
		//read entire file into string fileContent and and put it into fileWords vector
		fileContent.assign(istreambuf_iterator<char>(infile), istreambuf_iterator<char>());
		infile.close();
		fileContent = pp.remove_extra_space(fileContent);
		vector<string> fileWords = pp.word_tokenize(fileContent);
		allTrainingDocs.push_back(fileWords);  //add the tokenized file to the vector
	}
	
	vector<int> wordCount;
	int allTrainingDocsSize = allTrainingDocs.size();
	for (int i = 0; i < vocabSize; i++) {
		int numTimesAppeared = 0;
		for (int j = 0; j < allTrainingDocsSize; j++) {
			int allTrainingDocsJSize = allTrainingDocs[j].size();
			for (int k = 0; k < allTrainingDocsJSize; k++) {
				if (stopWords[i] == allTrainingDocs[j][k])
					numTimesAppeared++;
			}
		}
		wordCount.push_back(numTimesAppeared);
	}
	
	vector<pair<int, string> > featureCurve;
	vector<string> trainingNames2 = glob(directoryName + letter + "train*");
	vector<string> sampleNames2 = glob(directoryName + letter + "sample*");
	
	for (int i = 0; i < vocabSize; i++) {
		auto p1 = make_pair(wordCount[i], stopWords[i]);
		featureCurve.push_back(p1);
	}
	
	//sort the vector from greatest to least
	sort(featureCurve.rbegin(), featureCurve.rend());
	

	vector<string> includedFeatures;
	map <int, double> prior2;
	map <string, map<int, double>> condprob2;	
	for (int i = 10; i < vocabSize; i += 10) {
		includedFeatures.clear();
		
		for (int j = 0; j < i; j++)
			includedFeatures.push_back(featureCurve[j].second);
		
		//initialize a new object
		BernoulliNaiveBayes BNB2 = BernoulliNaiveBayes(trainingNames2, sampleNames2, includedFeatures, authorIDs, pp, letterPos);
		
		//clear the prior and condprob maps
		prior2.clear();
		condprob2.clear();
		
		BNB2.train(prior2, condprob2);
		
		predicted.clear();
		for (int j = 0; j < sampleNames2.size(); j++) {
			predicted.push_back(BNB2.test(prior2, condprob2, sampleNames2[j]));
		}
		
		//calculate the accuracy
		numCorrect = 0;
		for (int j = 0; j < actual.size(); j++) {
			if (actual[j] == predicted[j])
				numCorrect++;
		}
	
		cout << endl << i << " features give " << numCorrect/double(actual.size()) * 100 << " % " << "accuracy.";
	
	}
		includedFeatures.clear();
		
		for (int j = 0; j < 423; j++)
			includedFeatures.push_back(featureCurve[j].second);
		
		//initialize a new object
		BernoulliNaiveBayes BNB2 = BernoulliNaiveBayes(trainingNames2, sampleNames2, includedFeatures, authorIDs, pp, letterPos);
		
		//clear the prior and condprob maps
		prior2.clear();
		condprob2.clear();
		
		BNB2.train(prior2, condprob2);
		
		predicted.clear();
		for (int j = 0; j < sampleNames2.size(); j++) {
			predicted.push_back(BNB2.test(prior2, condprob2, sampleNames2[j]));
		}
		
		//calculate the accuracy
		numCorrect = 0;
		for (int j = 0; j < actual.size(); j++) {
			if (actual[j] == predicted[j])
				numCorrect++;
		}
	
		cout << endl << "423" << " features give " << numCorrect/double(actual.size()) * 100 << " % " << "accuracy.";
	cout << endl;		
    return 0;
}















