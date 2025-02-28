# V2XSBESCEMBEDDED
The embedded part of the V2X paper for SBESC 2024 

## ANN
Put the X_test.txt, Y_test.txt, and weights you got from the exporting folder from Python in the exported/ANN folder.
gcc -o test_ann test_ann.c -lm
time ./test_ann
Get the output time in seconds and divide by the number of samples in test data to get the inference time average.

## Random Forest
Put the X_test.txt, Y_test.txt, rf1.h and rf2.h files you got from the exporting folder from Python in the exported/RF folder.
gcc -o rf test_random_forest.c -I/exported/RF -lm
time ./rf
Get the output time in seconds and divide by the number of samples in test data to get the inference time average.

## XGBoost

Put the X_test.txt, Y_test.txt, and xgb_model.txt you got from the exporting folder from Python in the exported/XGBoost folder.