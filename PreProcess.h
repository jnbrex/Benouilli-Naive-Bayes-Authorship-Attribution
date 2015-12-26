#include <vector>
#include <string>
#include <string.h>
#include <sstream>
#include <map>
using namespace std;

class PreProcess{

public:

    PreProcess(){}

    // Function to print the confusion matrix.
    // Argument 1: "actual" is a list of integer class labels, one for each test example.
    // Argument 2: "predicted" is a list of integer class labels, one for each test example.
    // "actual" is the list of actual (ground truth) labels.
    // "predicted" is the list of labels predicted by your classifier.
    // "actual" and "predicted" MUST be in one-to-one correspondence.
    // That is, actual[i] and predicted[i] stand for testfile[i].
    void printConfMat(vector<int>actual, vector<int>predicted){
        vector<int> all_labels;
        assert(actual.size() == predicted.size());
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++)
            all_labels.push_back((*i));
        for (vector<int>::iterator i = predicted.begin(); i != predicted.end(); i++)
            all_labels.push_back((*i));
        sort( all_labels.begin(), all_labels.end() );
        all_labels.erase( unique( all_labels.begin(), all_labels.end() ), all_labels.end() );
        map<pair<int,int>, unsigned> confmat;  // Confusion Matrix
        int itt = 0;
        for (vector<int>::iterator i = actual.begin(); i != actual.end(); i++){
            int a = (*i);
            pair<int, int> pp = make_pair(a, predicted[itt]);
            if (confmat.find(pp) == confmat.end()) confmat[pp] = 1;
            else confmat[pp] += 1;
            itt++;
        }
        cout << "\n\n";
        cout << "0 ";  // Actual labels column (aka first column)
        vector<int> tmp_labels;
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label2 = (*i);
            cout << label2 << " ";
            tmp_labels.push_back(label2);
        }
        cout << "\n";
        for (vector<int>::iterator i = all_labels.begin(); i != all_labels.end(); i++){
            int label = (*i);
            cout << label << " ";
            for (vector<int>::iterator i2 = tmp_labels.begin(); i2 != tmp_labels.end(); i2++){
                int label2 = (*i2);
                pair<int, int> pp = make_pair(label, label2);
                if (confmat.find(pp) == confmat.end()) cout << "0 ";
                else cout << confmat[pp] << " ";
            }
            cout << "\n";
        }
    }

    // Function to remove leading, trailing, and extra space from a string.
    // Inputs a string with extra spaces.
    // Outputs a string with no extra spaces.
    string remove_extra_space(string str){
        string buf; // Have a buffer string
        stringstream ss(str); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        const char* const delim = " ";
        ostringstream imploded;
        copy(tokens.begin(), tokens.end(), ostream_iterator<string>(imploded, delim));
        return imploded.str();
    }

    // Tokenizer.
    // Input: string
    // Output: list of lowercased words from the string
    vector<string> word_tokenize(string input_string){
        string punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        string extra_space_removed = remove_extra_space(input_string);
        string punctuation_removed = "";
        for (unsigned i = 0; i < extra_space_removed.length(); i++) {
            char curr = extra_space_removed[i];
            if (punctuations.find(curr) == string::npos) punctuation_removed += curr;
        }
        transform(punctuation_removed.begin(), punctuation_removed.end(), punctuation_removed.begin(), ::tolower);
        string buf; // Have a buffer string
        stringstream ss(punctuation_removed); // Insert the string into a stream
        vector<string> tokens; // Create vector to hold our words
        while (ss >> buf) tokens.push_back(buf);
        return tokens;
    }

};
