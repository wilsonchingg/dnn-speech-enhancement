# Refactoring #

# Python Version #
Python 2.6+ / Python 3.5+
To run any scripts from the test/full_system_evaluation, Python 2.6+ must be required
# Installing dependencies #
run the following command from the project root directory
``` pip install -r requirements.txt ```

PESQ software: get the PESQ software from http://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en and run the following command in the software's source directory:
```gcc -o pesq *.c```
Place the binary file in any $PATH directory

# Acquire Full Dataset #
Download LibriSpeech dataset from http://www.openslr.org/12/ and place the uncompressed files to the appropriate directory

Download the DEMAND dataset from https://datashare.is.ed.ac.uk/handle/10283/2791 and place the uncompressed files to the appropriate directory

If a full dataset is used, search for TODO in the project folder and replace the parameter values accordingly

# Generating Training Dataset #
```
cd preprocessing
python generate_dataset.py
```
# Train a Neural Network (Example) #
```
cd train/mfcc_sequence
python topology_train.py
```
# Test a Neural Network (Example) #
```
cd test
python vanilla_net_test.py
```
# Analyze the Training Result #
run any python script from the analyze/plot directory
