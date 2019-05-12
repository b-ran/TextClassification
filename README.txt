Step 1: Install viritualenv package via:
	py -m pip install --user virtualenv

Step 2: Create a viritualenv via:
	py -m venv env

Step 3: Active the virtual environment
	source env/Scripts/activate or source env/bin/activate

Step 4: Install the packages into the virtual environment
	pip install keras
	pip install tensorflow
	pip install nltk

(Make sure to have the virtual environment actived before running)

Step 5: Running Baseline
	py basline.py

Step 6: Running Baseline With New Dataset
	py basline-newdata.py

Step 7: Running Improved Baseline With New Dataset
	py improve-newdata.py