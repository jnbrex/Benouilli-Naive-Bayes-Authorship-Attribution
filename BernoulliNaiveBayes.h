#include "PreProcess.h"
#include <unordered_set>
using namespace std;

class BernoulliNaiveBayes{
    vector<string> trainingNames;
    vector<string> sampleNames;
    vector<string> vocab;
    unordered_set<int> all_labels;
    PreProcess p;
	int letterPos;
public:
    BernoulliNaiveBayes(vector<string> trainingNames_in, vector<string> sampleNames_in, vector<string> vocab_in, unordered_set<int>& all_labels_in, PreProcess pp, int letterPos_in){
        trainingNames = trainingNames_in;
        sampleNames = sampleNames_in;
        vocab = vocab_in;
        all_labels = all_labels_in;
        p = pp;
        letterPos = letterPos_in;
    }

    // Define Train function
    void train(map <int, double>& prior, map <string, map<int, double>>& condprob) {
    	//this is the vocab vector								//V <- extractVocabulary(D)
    	double numDocs = trainingNames.size();					//N <- countDocs(D)
    	vector<int> Dc;
    	int vocabSize = vocab.size();	
    	for (int i = 1; i < all_labels.size() + 1; i++) {			//for each c E C
			double Nc = 0;										//do Nc <- CountDocsInClass(D,c)
			Dc.clear();
    		for (int j = 0; j < trainingNames.size(); j++) {
    			if (i == stoi(trainingNames[j].substr(letterPos + 1, 2))) {
    				Nc++;
    				Dc.push_back(j);
    			}			
    		}
    		prior.emplace(i, Nc/numDocs);						//prior[c] <- Nc/N
    		
    		vector<int> numDocsWithWord(vocab.size(), 0); //a vector to keep track of found words
    		for (int j = 0; j < Dc.size(); j++) {
    			ifstream infile;
    			infile.open(trainingNames[Dc[j]]);
    			string fileContent;
				//read entire file into string fileContent and and put it into fileWords vector
				fileContent.assign(istreambuf_iterator<char>(infile), istreambuf_iterator<char>());
				infile.close();
				fileContent = p.remove_extra_space(fileContent);
				vector<string> fileWords = p.word_tokenize(fileContent);
				
				//create an unordered set and fill it with file words
				unordered_set<string> fileWordSet;
				int fileWordsSize = fileWords.size();
				for (int k = 0; k < fileWordsSize; k++)
					fileWordSet.emplace(fileWords[k]);
    			for (int k = 0; k < vocabSize; k++) {			//for each t E V
    				//do Nct <- CountDocsInClassContainingTerm(D,c,t)
    				if (fileWordSet.find(vocab[k]) != fileWordSet.end())
    					numDocsWithWord[k]++;
    			}
    		}
    		//iterate through each vocab word	
			for (int j = 0; j < vocabSize; j++)
				//condprob[t][c] <- (Nct + 1)/(Nc + 2)
				condprob[vocab[j]][i] = (numDocsWithWord[j] + 1)/(Nc + 2);
		}
																//return V, prior, condprob
    }

    // Define Test function
    int test(map <int, double>& prior, map <string, map<int, double>>& condprob, string docName){
    	//Vd <- ExtractTermsFromDoc(V,d)
    	ifstream infile;
    	infile.open(docName);
		string fileContent;
		//read entire file into string fileContent and then put it into fileWords vector
		fileContent.assign(istreambuf_iterator<char>(infile), istreambuf_iterator<char>());
		infile.close();
		fileContent = p.remove_extra_space(fileContent);
		vector<string> fileWords = p.word_tokenize(fileContent);
		//create an unordered set and fill it with file words
		unordered_set<string> fileWordSet;
		int fileWordsSize = fileWords.size();
		for (int k = 0; k < fileWordsSize; k++)
			fileWordSet.emplace(fileWords[k]);
    	
    	//initialize vector to hold scores for each author
    	vector<double> scores(all_labels.size(), 0);
    	int vocabSize = vocab.size();
    	for (int i = 1; i < all_labels.size() + 1; i++) {		//for each c E C
    		scores[i - 1] = log2(prior[i]);							//do score[c] <- log prior[c]
    		for (int j = 0; j < vocabSize; j++)	{				//for each t E V
    			if (fileWordSet.find(vocab[j]) != fileWordSet.end())	//do if t E Vd
					scores[i - 1] += log2(condprob[vocab[j]][i]);
				else
					scores[i - 1] += log2(1 - condprob[vocab[j]][i]);
    		}
    	}
    	int scoresSize = scores.size();
    	double argmax = scores[0];
    	int author = 1;
    	//determine argmax
    	for (int i = 1; i < scoresSize; i++) {
    		if (scores[i] > argmax) {
    			argmax = scores[i];
    			author = i + 1;
    		}
    	}
    	
    	return author;											//return argmax c E C score[c]
    }  
};
