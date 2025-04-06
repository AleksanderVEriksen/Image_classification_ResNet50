from PIL import Image
from matplotlib.pyplot import imshow
import pandas
import matplotlib.pylab as plt
import os
import asyncio
import shutil
import skillsnetwork
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input

# Get the environment variable for the dataset path
dataset_path = os.getenv("RESNET50_PATH")

# Created Functions
async def download_data():
        # Download the dataset using skillsnetwork
        await skillsnetwork.prepare(
            "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", 
            path = dataset_path, 
            overwrite=True
            )
def download():
        # Set a valid temporary directory for Windows
        tmp_dir ="C:/tmp"

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        os.environ["TEMP"] = tmp_dir

        # Create directory if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Clear tmp directory
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        # Download the dataset
        asyncio.run(download_data())
def prepare_data():
        # Split class data
        directory = "resources\\data"

        negative_path = os.path.join(directory, "Negative")
        positive_path = os.path.join(directory, "Positive")

        # Rename files to avoid confusion
        def rename_files(path, prefix):
            for filename in os.listdir(path):
                os.rename(os.path.join(path, filename), os.path.join(path, f"{prefix}_{filename}"))

        # Check if files are present in the directories
        if os.path.exists(negative_path) and os.path.exists(positive_path):
            if len(os.listdir(negative_path)) == 0 and len(os.listdir(positive_path)) == 0:
                print("No files found in the directories.")
            else:
                rename_files(negative_path, "Negative")
                rename_files(positive_path, "Positive")
                # Check the number of images in each directory
                print("Number of images in Negative directory:", len(os.listdir(negative_path)))
                print("Number of images in Positive directory:", len(os.listdir(positive_path)))

                negative_files = [os.path.join(negative_path, f) for f in os.listdir(negative_path) if f.endswith('.jpg')]
                negative_files = sorted(negative_files)
                positive_files = [os.path.join(positive_path, f) for f in os.listdir(positive_path) if f.endswith('.jpg')]
                positive_files = sorted(positive_files)

            # Create train and test directories
            def create_directory(path):
                if not os.path.exists(path):
                    os.makedirs(path)
                    os.makedirs(os.path.join(path, "Negative"))
                    os.makedirs(os.path.join(path, "Positive"))
                return path

            # Create train, test, and validation directories if they don't exist
            train_dir = create_directory(dataset_path+"\\train")
            test_dir = create_directory(dataset_path+"\\test")
            validation_dir = create_directory(dataset_path+"\\validation")
            print("Moving files to train, test, and validation directories...")
            # Function to move files
            def move_files(files, dir1, dir2, dir3, class_name):
                if files:
                    # Move 90% of the images to the train directory
                    train_split = int(len(files)*0.90)
                    train_files = files[:train_split]
                    for file in train_files:
                        try:
                            shutil.move(file, os.path.join(dir1, class_name, os.path.basename(file)))
                        except FileExistsError:
                            print(f"File already exists: {os.path.join(dir1, class_name, os.path.basename(file))}")
                    
                    # Move 25% of train images to the validation directory
                    train_dir_files = os.listdir(os.path.join(dir1, class_name))
                    validation_split = int(len(train_dir_files)*0.1)
                    validation_files = train_dir_files[:validation_split]
                    for file in validation_files:
                        file_path = os.path.join(dir1, class_name, file)
                        try:
                             shutil.move(file_path, os.path.join(dir2, class_name, os.path.basename(file)))
                        except FileExistsError:
                            print(f"File already exists: {os.path.join(dir2, class_name, os.path.basename(file))}")
                    
                    # Move 10% of the images to the test directory
                    test_files = files[train_split:]
                    for file in test_files:
                        try:
                            shutil.move(file, os.path.join(dir3, class_name, os.path.basename(file)))
                        except FileExistsError:
                            print(f"File already exists: {os.path.join(dir3, class_name, os.path.basename(file))}")
            # Check if train, validation, and test directories are empty
            def are_directories_empty(*dirs):
                for directory in dirs:
                    negative_dir = os.path.join(directory, "Negative")
                    positive_dir = os.path.join(directory, "Positive")
                    if os.path.exists(negative_dir) and os.listdir(negative_dir):
                        return False
                    if os.path.exists(positive_dir) and os.listdir(positive_dir):
                        return False
                return True
            # Move files to train, validation, and test directories if they are not empty
            if are_directories_empty(train_dir, validation_dir, test_dir):
                move_files(negative_files, train_dir, validation_dir, test_dir, class_name="Negative")
                move_files(positive_files, train_dir, validation_dir, test_dir, class_name="Positive")
                print("Data preparation completed.")
            # Print the number of images in each directory
            print("Number of images in train Negative directory:", len(os.listdir(os.path.join(train_dir, "Negative"))))
            print("Number of images in train Positive directory:", len(os.listdir(os.path.join(train_dir, "Positive"))))
            print("\nNumber of images in test Negative directory:", len(os.listdir(os.path.join(test_dir, "Negative"))))
            print("Number of images in test Positive directory:", len(os.listdir(os.path.join(test_dir, "Positive"))))
            print("\nNumber of images in validation Negative directory:", len(os.listdir(os.path.join(validation_dir, "Negative"))))
            print("Number of images in validation Positive directory:", len(os.listdir(os.path.join(validation_dir, "Positive"))))
        else:
            # Get the dataset paths
            train_dir = dataset_path+"\\train"
            test_dir = dataset_path+"\\test"
            validation_dir = dataset_path+"\\validation"
            
        # Remove empty directories
        if os.path.exists(negative_path):
            os.rmdir(negative_path)
        if os.path.exists(positive_path):
            os.rmdir(positive_path)

        return train_dir, test_dir, validation_dir
# Define the data_gen parameters
def data_generation(directory, shuffle_data=True):
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        data_gen = data_generator.flow_from_directory(
            directory,
            batch_size=4,
            class_mode='categorical',
            seed=24,
            target_size=(224, 224),
            shuffle=shuffle_data,
        )
        return data_gen
def plot_image_batch(generator):
        # Plot images from the generator
        first_batch_images = generator.next()[0]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        ind = 0
        for ax1 in axs:
            for ax2 in ax1:
                image_data = first_batch_images[ind].astype(np.uint8)
                ax2.imshow(image_data)
                ind += 1
        fig.suptitle("First Batch of Images")
        plt.show()
# Define the model
def create_model(num_classes):
    model = Sequential()

    # Add ResNet50 model
    model.add(ResNet50(
        include_top=False,
        pooling="avg",
        weights="imagenet",
    ))
    model.add(Dense(num_classes, activation="softmax"))
    model.layers[0].trainable = False

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
# Fit the model
def fit_model(model, epochs ,training_data, validation_data):

        save_path = "classifier_resnet_model.h5"
        # Check if the model already exists
        Checkpoints = keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        Callback_list = [Checkpoints]
        # Check if the model already exists
        train_steps = 100 // training_data.batch_size # Max is 10 000
        validation_steps = 100 // validation_data.batch_size # Max is 10 000

        history = model.fit(
            training_data,
            steps_per_epoch=validation_steps,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps, 
            callbacks=Callback_list
        )
        return history

# Plot the training history
def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Make predictions on new images
def predict_image(model, image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prediction = model.predict(img)
    return prediction

# Plot 
def plot_prediction(image_path, prediction):
    img = Image.open(image_path).resize((224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted class: {(prediction)}")
    plt.show()


def generate_unique_filename(base_name, extension):
            counter = 1
            while True:
                model_name = f"{base_name}_{counter}.{extension}"
                if not os.path.exists(model_name):
                    return model_name
                counter += 1
# Load the project
def load_project(Training, Loading, Evaluation, Prediction, remove_data=False):
    # Load dataset and create directory if resources/data does not exist
    if not os.path.exists('resources/data') or (os.path.exists('resources/data') and len(os.listdir('resources/data')) == 0):
        print("Downloading the dataset...")
        download()
    # Prepare the dataset by splitting the Positive and Negative files into training, test and validation directories
    train_dir, test_dir, validation_dir = prepare_data()
    # Create image generators for train, test and validation
    train_generator = data_generation(train_dir)
    test_generator = data_generation(test_dir, shuffle_data=False)
    validation_generator = data_generation(validation_dir)
    # Get classes
    num_classes = len(train_generator.class_indices)
    if Loading:
        # Load model -- Trained with priority
        model_name = 'classifier_resnet_model.h5'
        model_name_trained = 'classifier_resnet_model_TRAINED.h5'
        if os.path.exists(model_name_trained):
            print("Loading the trained model...")
            try:
                model = keras.models.load_model(model_name_trained)
            except Exception as e:
                print(f"Failed to load the trained model: {e}")
        else:
            if os.path.exists(model_name):
                print("Loading the model...")
                model = keras.models.load_model(model_name)
    else:
        # Create model
        model = create_model(num_classes)
        # Save the model before training
        model.save('classifier_resnet_model.h5')
    # Fit the model
    if Training:
        history = fit_model(model, 1 , train_generator, validation_generator)
        model_name = "classifier_resnet_model_TRAINED.h5"
        print("Training completed.")
        # Save the model after training
        print("saving model...")
        model.save(model_name)
        # Plot the training history
        plot_history(history)
    if Evaluation:
        # Evaluate the model
        print("Evaluating the model...")
        score = model.evaluate(test_generator, steps=len(test_generator))
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    if Prediction:
        # Prediction
        image_path = "resources/data/test/Positive/Positive_18001_1.jpg"
        prediction = predict_image(model, image_path)
        if np.argmax(prediction) == 0:
             prediction = "Negative"
        else:
             prediction = "Positive"
        print("Predicted class:", prediction)
        # Plot the prediction
        plot_prediction(image_path, prediction)
        # Adjust steps_per_epoch for the test generator to 2500
        step = 2500 # Max is 10 000
        steps_per_epoch = step // test_generator.batch_size
        predictions = model.predict(test_generator, steps=steps_per_epoch, verbose=2)
        # Get the predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        # Get the actual classes
        actual_classes = test_generator.classes[:step]
        # Plot the first 5 predicted and actual classes
        for i in range(5):
            # Get the file path from the test generator
            image_path = os.path.join(
                "resources/data/test",
                test_generator.filenames[i]
            )
            # Ensure the file exists
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            # Get the predicted and actual class labels
            prediction = "Negative" if predicted_classes[i] == 0 else "Positive"
            actual = "Negative" if actual_classes[i] == 0 else "Positive"
            # Print the result
            print(f"Actual class: {actual}, Predicted class: {prediction}")
            # Plot the prediction
            plot_prediction(image_path, prediction)
        
        # Get the confusion matrix
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(actual_classes, predicted_classes)
        print("Confusion Matrix:")
        print(confusion_matrix)
        # Plot the confusion matrix
        def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes.keys(), rotation=45)
            plt.yticks(tick_marks, classes.keys())
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()
        plot_confusion_matrix(confusion_matrix, classes=test_generator.class_indices, title="Confusion Matrix")
        # Save the confusion matrix
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        # Calculate metrics
        accuracy = accuracy_score(actual_classes, predicted_classes)
        precision = precision_score(actual_classes, predicted_classes, zero_division=0.0)
        recall = recall_score(actual_classes, predicted_classes, zero_division=0.0)
        f1 = f1_score(actual_classes, predicted_classes, zero_division=0.0)

        # Print metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        if remove_data:
            # Delete the downloaded dataset
            shutil.rmtree("resources/data")
            # Delete the temporary directory
            shutil.rmtree("C:/tmp")

# Main function to run the project
if __name__ == "__main__":
    # Run the project
    # If model exists, load it
    if os.path.exists('classifier_resnet_model_TRAINED.h5'):
        load_project(Training=False, Loading=True, Evaluation=False, Prediction=True)
    else:
        print("Model does not exist. Creating a new model for training..")
        try:
             load_project(Training=True, Loading=False, Evaluation=True, Prediction=True)
        except OSError:
             shutil.rmtree("resources/data")


