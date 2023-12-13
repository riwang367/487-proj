# EECS 487 Final Project

## Setup
Clone this repository to your local machine

	git clone https://github.com/riwang367/487-proj.git

Download `.joblib` files from Drive and add them to the `website/fandomclf` folder in your local directory
	
- [5000_clf.joblib](https://drive.google.com/file/d/18lQkaqY-lx6X8VQq_wJCOtctwXa4aPI6/view?usp=drive_link)
- [5000_vectorizer.joblib](https://drive.google.com/file/d/1kRNJc0QzxHYENE_MZafw7rJSHLEmRgfM/view?usp=drive_link)

Change to the website directory and create a virtual environment in the website folder.
   
    cd website
    python3 -m venv env

Activate the virtual environment

    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    pip install -e .

`chmod` the bash script:

    chmod +x ./bin/clf_run

## Running the app
Change directory to the website folder:
	
 	cd website

Run the local server:

	source env/bin/activate
    ./bin/clf_run
    

## Resources
[EECS 485 Virtual Environment Setup Tutorial](https://eecs485staff.github.io/p1-insta485-static/setup_virtual_env.html)

[All saved classifiers](https://drive.google.com/drive/folders/1m1KvKKQivOn3Wi9ba6jw_okkxD-lnoym?usp=sharing)
- Feedforward neural network classifier/vectorizer trained on 40,000 posts (5,000 per subreddit)
- Naive Bayes classifier/vectorizer trained on 40,000 posts
- Feedforward neural network classifier trained on 8,000 posts (1,000 per subreddit)