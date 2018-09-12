# Note #
This repository is intended as a reference for ideal ratio mask dnn implementation

It is based on my university dissertation. Please send me an email if you would like to know more about it.

# Python Version #
Python 3 (Python 2 also works with some modifications, search TODO in the repository)

# Installing dependencies #
run the following command from the project root directory
``` pip install -r requirements.txt ```

PESQ software: get the PESQ software from http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en and run the following command in the software's source directory:
```gcc -o pesq *.c```
Place the binary file in any $PATH directory

# Acquire Full Dataset #
Download LibriSpeech dataset from http://www.openslr.org/12/ and place the uncompressed files to the appropriate directory

Download the DEMAND dataset from https://datashare.is.ed.ac.uk/handle/10283/2791 and place the uncompressed files to the appropriate directory

# Generating Training Dataset #
```
cd preprocessing
python generate_dataset.py
```
# Train a Neural Network (Example) #
```
cd train
python residual_train.py
```
# Test a Neural Network (Example) #
```
cd test
python test_snr_performance.py
```
