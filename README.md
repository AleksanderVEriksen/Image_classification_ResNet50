# Classification of concrete cracks using the pre-trained ResNet50

Classification of concrete cracks is very important for monitoring health of buildings. Severe cracks might conclude in a building falling apart.
This projects involves classification of different images showing cracks or not. The images are divided into Negative and Positive. Negative represent no cracks, while Positive represent cracks.

## How to run

Important notice!.
Make sure to have D:/tmp or update line 160 to a designated folder. Also an environmental variable RESNET50_PATH have been set in order to download the dataset from skillsnetwork. The path points towards the resource folder made from the download.

### Install dependencies

Open the project and go to the Image_Classification_RESNET50 folder.
Then open the terminal and type this to enter virtual environment and install the required dependencies:
``Python
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
``

### Run the API server

To launch the API server, type this into the terminal to start the FastAPI server:
``python
uvicorn Concrete_crack_classification:app --host 127.0.0.1 --port 8000 --reload
``

Either use the UI you met when opening ``http://127.0.0.1:8000`` in a browser, or use the terminal by following below:

The server has these endpoints to see status, predict and train a model within the terminal:

```Python
Invoke-RestMethod -Uri http://127.0.0.1:8000/health

Invoke-WebRequest -Uri http://127.0.0.1:8000/predict -Method Post -Form @{ file = Get-Item "resources/data/test/Positive/Positive_XXXX.jpg" }

Invoke-RestMethod -Uri http://127.0.0.1:8000/train -Method Post
```

To see the 5 predicted images or the confusion matrix, go to:

* http://127.0.0.1:8000/confusion-matrix
* http://127.0.0.1:8000/predictions-image

### Run code in terminal

If you want to just run the code and train a model through python without API, use the following command in the terminal:

``python
python Concrete_crack_RESNET50.py
``

## Prediction of 5 random images

![predictions](https://github.com/user-attachments/assets/663c3ddc-b6b5-4cbc-84b7-186820c09062)

### Confusion matrix

![confusion_matrix](https://github.com/user-attachments/assets/a31f60cb-56f4-495e-8fb3-bc0bd0a99791)

### Source for dataset

[Concrete dataset](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip)
