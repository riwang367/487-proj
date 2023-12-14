# EECS 487 Final Project

## Setup
Clone this repository to your local machine:

	git clone https://github.com/riwang367/487-proj.git

Download the following `.joblib` files from Drive and add them to the `website/fandomclf` folder in your local directory:
	
- [1000_ffnn.joblib](https://drive.google.com/file/d/1TKnC5EeHst9cSA37WW1KFrBdhoIpxEbf/view?usp=drive_link)
- [1000_ffnn_vectorizer.joblib](https://drive.google.com/file/d/1-88JvU7-LTTmSPuORxyRUO5Ip9gQJH2_/view?usp=sharing)


Update Python packages to ensure everything is up-to-date for the virtual environment:

    sudo apt-get update
    sudo apt-get install python3 python3-pip python3-venv python3-wheel python3-setuptools


Change to the website directory and create a virtual environment in the website folder:

    cd website
    python3 -m venv env

`chmod` the bash script:

    chmod +x ./bin/clf_run

Activate the virtual environment and install/upgrade packages:

    source env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt


## Running the app
Change directory to the website folder:
	
 	cd website

Run the local server:

	source env/bin/activate
    ./bin/clf_run
    

## Resources
[EECS 485 Virtual Environment Setup Tutorial](https://eecs485staff.github.io/p1-insta485-static/setup_virtual_env.html)

[All saved classifiers](https://drive.google.com/drive/folders/1m1KvKKQivOn3Wi9ba6jw_okkxD-lnoym?usp=sharing)
- Feedforward neural network classifier/vectorizer trained on 40,000 posts (5000 posts per subreddit): 5000_ffnn.joblib and 5000_ffnn_vectorizer.joblib
- Naive Bayes classifier/vectorizer trained on 5000 posts per subreddit: 5000_nb.joblib and 5000_nb_vectorizer.joblib
- Feedforward neural network classifier/vectoriser trained on 8,000 posts (1,000 per subreddit) and used in our website: 1000_ffnn.joblib and 1000_ffnn_vectorizer.joblib 
- Naive Bayes classifier/vectorizer trained on 8,000 posts (1,000 per subreddit): 1000_nb.joblib and 1000_nb_vectorizer.joblib