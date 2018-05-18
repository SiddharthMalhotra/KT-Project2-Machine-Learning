Assignment 2: Knowledge Technologies
Identifying Tweets with Adverse Drug Reactions

tokens.txt
	A feature engineered token file. In this project we are using trigrams as reference. 

n-gram.py
	Generates a CSV based on the value of n mentioned as the input 
	Takes the .txt file containing raw tweets as the input 
	Runs over each line of the token file and looks for the corresponding n-gram frequency 
	Outputs a feature vector space

New_dev.arff & New_train.arff & New_test.arff
	We generate these by first removing the header and keeping it seperately. 
	Conversting out arff file into a csv file 
	Appending our csv generated from n-gram.py 
	Converting it back to arff
	Adding our headers back to the file 
	We maintain a consistent representation for all of our tweet files 

Output Predictions Directory 
	The outputs generated which regards to 'prediction' on TEST 
	We run multiple algorithms in context to our result analysis 

Spyder-Perceptron (An experiment)
	An experimentation over the perceptron algorithm without the appending of given attributes 
	The file uses sklearn and numpy 
	We try to learn the process of machine learning and perfrom splitting of train and test data 