from PIL import Image
import matplotlib.pylab as plt
import os
# Set TensorFlow environment variables before importing TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # Disable oneDNN custom ops
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # Suppress INFO logs
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # Prefer deterministic behavior
import asyncio
import shutil
import skillsnetwork
import numpy as np
import tensorflow as tf
# Prefer TensorFlow Keras imports for consistency
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException

from starlette.responses import FileResponse, JSONResponse, HTMLResponse
import io

# Globals for API serving
CLASS_NAMES = ["Negative", "Positive"]
MODEL = None

def _ensure_dirs():
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    if not os.path.exists("resources"):
        os.makedirs("resources")
    if not os.path.exists(os.path.join("resources", "data")):
        os.makedirs(os.path.join("resources", "data"))

_ensure_dirs()

def _load_or_init_model():
    global MODEL
    model_name_trained = 'classifier_resnet_model_TRAINED.keras'
    model_name = 'classifier_resnet_model.keras'
    try:
        if os.path.exists(model_name_trained):
            try:
                MODEL = tf.keras.models.load_model(model_name_trained, compile=False)
            except Exception as e:
                print(f"Failed to load trained model; falling back: {e}")
                # Try weights-only fallback
                try:
                    MODEL = create_model(len(CLASS_NAMES))
                    MODEL.load_weights(model_name_trained)
                    print("Loaded trained weights into freshly built model.")
                except Exception as e2:
                    print(f"Failed to load trained weights: {e2}")
                    MODEL = None
        if MODEL is None and os.path.exists(model_name):
            try:
                MODEL = tf.keras.models.load_model(model_name, compile=False)
            except Exception as e:
                print(f"Failed to load base model; will recreate: {e}")
                # Try weights-only fallback for base model
                try:
                    MODEL = create_model(len(CLASS_NAMES))
                    MODEL.load_weights(model_name)
                    print("Loaded base weights into freshly built model.")
                except Exception as e2:
                    print(f"Failed to load base weights: {e2}")
                    MODEL = None
        if MODEL is None:
            MODEL = create_model(len(CLASS_NAMES))
            try:
                MODEL.save(model_name)
            except Exception as e:
                print(f"Warning: could not save base model: {e}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        MODEL = create_model(len(CLASS_NAMES))

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_or_init_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
        html = """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>Concrete Crack Classification API</title>
            <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
                .container { max-width: 900px; margin: 40px auto; padding: 24px; }
                h1 { margin: 0 0 8px; font-size: 28px; }
                p { margin: 0 0 16px; color: #cbd5e1; }
                .actions { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }
                .btn { display: inline-block; padding: 10px 14px; border-radius: 8px; text-decoration: none; font-weight: 600; }
                .primary { background: #3b82f6; color: white; }
                .secondary { background: #334155; color: #e2e8f0; }
                .card { background: #0b1220; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }
                input[type=file] { display: block; margin: 10px 0 16px; }
                button { padding: 10px 14px; border-radius: 8px; border: none; font-weight: 600; background: #22c55e; color: #052e1a; cursor: pointer; }
                pre { background: #0b1220; padding: 12px; border-radius: 8px; overflow-x: auto; }
                a { color: inherit; }
            </style>
        </head>
        <body>
            <div class=\"container\"> 
                <h1>Concrete Crack Classification API</h1>
                <p>Use the quick actions below or upload an image to get a prediction.</p>
                <div class=\"actions\">
                    <a class=\"btn primary\" href=\"/docs\">Open API Docs</a>
                    <a class=\"btn secondary\" href=\"/health\">Health</a>
                    <a class=\"btn secondary\" href=\"/confusion-matrix\">Confusion Matrix</a>
                    <a class=\"btn secondary\" href=\"/predictions-image\">Predictions Image</a>
                    <a class=\"btn secondary\" href=\"/train\">Start Training</a>
                </div>
                <div class=\"card\">
                    <h2 style=\"margin-top:0\">Predict an Image</h2>
                    <form id=\"predictForm\" action=\"/predict\" method=\"post\" enctype=\"multipart/form-data\">
                        <input type=\"file\" name=\"file\" accept=\"image/*\" required />
                        <button type=\"submit\">Predict</button>
                    </form>
                    <div id=\"result\" style=\"margin-top:12px\"></div>
                </div>
                <p style=\"margin-top:18px\">Tip: for detailed controls, open the <a href=\"/docs\">Swagger UI</a>.</p>
            </div>
            <script>
                    const form = document.getElementById('predictForm');
                    const fileInput = form.querySelector('input[name="file"]');
                    form.addEventListener('submit', async (e) => {
                        e.preventDefault();
                        const fd = new FormData(form);
                        const resEl = document.getElementById('result');
                        resEl.innerHTML = 'Predicting...';
                        try {
                            const resp = await fetch('/predict', { method: 'POST', body: fd });
                            const text = await resp.text();
                            try {
                                const json = JSON.parse(text);
                                // Preview the uploaded image and show predicted label
                                const file = fileInput.files && fileInput.files[0];
                                let imgHtml = '';
                                if (file) {
                                    const url = URL.createObjectURL(file);
                                    imgHtml = `<img src="${url}" alt="Uploaded Image" style="max-width:300px;border-radius:8px;display:block;" />`;
                                    // Revoke after a short delay to allow rendering
                                    setTimeout(() => URL.revokeObjectURL(url), 3000);
                                }
                                const label = (json.predicted_label ?? '').toString();
                                const probs = json.probabilities ? `<pre>${JSON.stringify(json.probabilities, null, 2)}</pre>` : '';
                                resEl.innerHTML = `
                                    <div style="margin-top:12px">
                                        ${imgHtml}
                                        <div style="margin-top:8px;font-weight:700">Predicted: ${label}</div>
                                        ${probs}
                                    </div>
                                `;
                            } catch {
                                // Fallback: non-JSON error or unexpected response
                                resEl.innerHTML = '<pre>' + text.replace(/</g,'&lt;') + '</pre>';
                            }
                        } catch (err) {
                            resEl.textContent = 'Error: ' + (err && err.message ? err.message : err);
                        }
                    });
                </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)


def _predict_from_bytes(image_bytes: bytes):
    if MODEL is None:
        _load_or_init_model()
    with Image.open(io.BytesIO(image_bytes)) as im:
        img = im.resize((224, 224))
        arr = np.array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        preds = MODEL.predict(arr)
        probs = preds[0].tolist()
        idx = int(np.argmax(preds, axis=1)[0])
        return {
            "predicted_index": idx,
            "predicted_label": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
            "probabilities": {CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i): float(p) for i, p in enumerate(probs)}
        }

def _train_task():
    try:
        load_project(Training=True, Loading=False, Evaluation=True, Prediction=False)
    finally:
        _load_or_init_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await file.read()
    try:
        results = _predict_from_bytes(data)
        html = """ \
        "<!DOCTYPE html>" \
        "<html><head><title>" \
        "Prediction Result" \
        "</title>" \
        "</head>" \
        "<img src={image} alt="Uploaded Image" style="max-width:300px;"/><br>" \
        "<h2>Prediction: {prediction}</h2>" \
        "<body>"
        """
        
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(_train_task)
    return {"status": "started"}

@app.get("/confusion-matrix")
def confusion_matrix_image():
    path = os.path.join("predictions", "confusion_matrix.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix not found")

@app.get("/predictions-image")
def predictions_image():
    path = os.path.join("predictions", "predictions.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Predictions image not found")

# Get the environment variable for the dataset path
dataset_path = os.getenv("RESNET50_PATH")
if dataset_path is None:
    dataset_path = os.path.join(os.getcwd(), "resources/data")

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
        tmp_dir ="D:/tmp"

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        #os.environ["TMPDIR"] = tmp_dir
        #os.environ["TMP"] = tmp_dir
        #os.environ["TEMP"] = tmp_dir

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
        directory = dataset_path

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
            train_dir = create_directory(os.path.join(dataset_path, "train"))
            test_dir = create_directory(os.path.join(dataset_path, "test"))
            validation_dir = create_directory(os.path.join(dataset_path, "validation"))
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
            train_dir = os.path.join(dataset_path, "train")
            test_dir = os.path.join(dataset_path, "test")
            validation_dir = os.path.join(dataset_path, "validation")
            
        # Remove empty directories
        if os.path.exists(negative_path):
            os.rmdir(negative_path)
        if os.path.exists(positive_path):
            os.rmdir(positive_path)

        return train_dir, test_dir, validation_dir
# Define the data_gen parameters
def data_generation(directory, shuffle_data=True):
        # Create a tf.data.Dataset from a directory structure
        ds = image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            batch_size=4,
            image_size=(224, 224),
            shuffle=shuffle_data,
            seed=24,
        )
        # Apply ResNet50 preprocessing to batches
        ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
def plot_image_batch(generator):
    # Plot images from the dataset (first batch)
    first_batch_images, _ = next(iter(generator))
    # Clip to [0, 255] and cast for display
    first_batch_images = tf.clip_by_value(first_batch_images, 0, 255).numpy().astype(np.uint8)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    ind = 0
    for ax1 in axs:
        for ax2 in ax1:
            image_data = first_batch_images[ind]
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
        input_shape=(224, 224, 3),
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

        save_path = "classifier_resnet_model.keras"
        # Check if the model already exists
        Checkpoints = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        Callback_list = [Checkpoints]
        # Determine number of batches from dataset cardinality
        train_steps = tf.data.experimental.cardinality(training_data).numpy()
        validation_steps = tf.data.experimental.cardinality(validation_data).numpy()

        history = model.fit(
            training_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps, 
            callbacks=Callback_list
        )
        return history

# Plot the training history
def plot_history(history):
    # Extract accuracy metrics robustly
    acc = history.history.get('accuracy') or history.history.get('categorical_accuracy') or []
    val_acc = history.history.get('val_accuracy') or history.history.get('val_categorical_accuracy') or []
    epochs = list(range(1, max(len(acc), len(val_acc)) + 1))

    plt.figure(figsize=(8, 5))
    if acc:
        plt.plot(epochs, acc, marker='o', label='Train')
    if val_acc:
        plt.plot(epochs, val_acc, marker='s', label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    # Save and show for visibility
    try:
        os.makedirs('predictions', exist_ok=True)
        plt.savefig(os.path.join('predictions', 'training_history.png'))
    except Exception:
        pass
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
def plot_predictions(image_paths, predictions):
    """
    Plots multiple images with their predictions.

    Args:
        image_paths (list): List of image file paths.
        predictions (list): List of predictions corresponding to the images.
    """
    # Ensure inputs are lists for consistent processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(predictions, str):
        predictions = [predictions]
    
    if len(image_paths) != len(predictions):
        raise ValueError("The number of images and predictions must match.")
    if len(image_paths) == 1 and len(predictions) == 1:
        cols = 1
    else:
        cols = 3  # Number of columns in the grid
    num_images = len(image_paths)

    rows = (num_images + cols - 1) // cols  # Calculate rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_images == 1:
        axes =[axes]  # If only one image, keep axes as a single element array
    else:
        axes = axes.flatten()  # Flatten the axes array for easy iteration
    for i, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        img = Image.open(image_path).resize((224, 224))
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Predicted: {prediction}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join("predictions", "predictions.png"))
    plt.show()
    

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
            plt.figure(figsize=(10, 8))  # Adjust the width and height as needed
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes.keys(), rotation=45)
            plt.yticks(tick_marks, classes.keys())
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join("predictions", "confusion_matrix.png"))
            plt.show()
            

def select_5_images_for_prediction(predicted_classes, actual_classes, test_dir):
            # Randomly select 5 images from test directory and predict
            import random
            image_paths = []
            predicted_labels = []
            actual_labels = []

            candidates = []
            for cls in ["Negative", "Positive"]:
                cls_dir = os.path.join(test_dir, cls)
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            candidates.append(os.path.join(cls_dir, f))

            if not candidates:
                print("No test images found for prediction preview.")
                return

            for image_path in random.sample(candidates, k=min(5, len(candidates))):
                image_paths.append(image_path)
                # Actual label inferred from path
                actual_labels.append(os.path.basename(os.path.dirname(image_path)))
                # Predict using the model on each image
                pred = predict_image(MODEL, image_path)
                idx = int(np.argmax(pred, axis=1)[0])
                predicted_labels.append("Negative" if idx == 0 else "Positive")
                print(f"Actual class: {actual_labels[-1]}, Predicted class: {predicted_labels[-1]}")

            plot_predictions(image_paths, predicted_labels)

# Load the project
def load_project(Training,Loading, Evaluation, Prediction, remove_data=False, epoch:int = 1):
    global MODEL
    # Load dataset and create directory if dataset_path does not exist
    if not os.path.exists(dataset_path) or (os.path.exists(dataset_path) and len(os.listdir(dataset_path)) == 0):
        print("Downloading the dataset...")
        download()
    # Prepare the dataset by splitting the Positive and Negative files into training, test and validation directories
    train_dir, test_dir, validation_dir = prepare_data()
    # Create image generators for train, test and validation
    train_generator = data_generation(train_dir)
    test_generator = data_generation(test_dir, shuffle_data=False)
    validation_generator = data_generation(validation_dir)
    # Get classes
    num_classes = len(getattr(train_generator, 'class_names', CLASS_NAMES))
    model = None
    if Loading:
        # Load model -- Trained with priority
        model_name = 'classifier_resnet_model.keras'
        model_name_trained = 'classifier_resnet_model_TRAINED.keras'
        if os.path.exists(model_name_trained):
            print("Loading the trained model...")
            try:
                model = tf.keras.models.load_model(model_name_trained, compile=False)
            except Exception as e:
                print(f"Failed to load the trained model: {e}")
                # Fallback: build architecture and load weights only
                try:
                    model = create_model(num_classes)
                    model.load_weights(model_name_trained)
                    print("Loaded trained weights into freshly built model.")
                except Exception as e2:
                    print(f"Failed to load trained weights: {e2}")
                    model = None
        if model is None and os.path.exists(model_name):
            print("Loading the base model...")
            try:
                model = tf.keras.models.load_model(model_name, compile=False)
            except Exception as e:
                print(f"Failed to load the base model: {e}")
                # Fallback: build architecture and load weights only
                try:
                    model = create_model(num_classes)
                    model.load_weights(model_name)
                    print("Loaded base weights into freshly built model.")
                except Exception as e2:
                    print(f"Failed to load base weights: {e2}")
                    model = None
    else:
        # Create model
        model = create_model(num_classes)
        # Save the model before training
        model.save('classifier_resnet_model.keras')

    # If still no model (e.g., load failures), create a fresh one
    if model is None:
        print("Creating a fresh model due to load failures...")
        model = create_model(num_classes)

    # Keep global MODEL in sync for downstream functions that rely on it
    MODEL = model
    # Fit the model
    if Training:
        history = fit_model(model, epoch, train_generator, validation_generator)
        model_name = "classifier_resnet_model_TRAINED.keras"
        print("Training completed.")
        # Save the model after training
        print("saving model...")
        model.save(model_name)
        # Plot the training history
        plot_history(history)
    if Evaluation and model is not None:
        # Evaluate the model
        print("Evaluating the model...")
        val_steps = tf.data.experimental.cardinality(test_generator).numpy()
        score = model.evaluate(test_generator, steps=val_steps)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    if Prediction and model is not None:
        print("Predicting on new images...")
        # Predict across entire test dataset
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        # Extract actual classes from dataset
        actual_classes = []
        for _, y in test_generator:
            actual_classes.extend(np.argmax(y.numpy(), axis=1))
        actual_classes = np.array(actual_classes)[:len(predicted_classes)]
        # Select and plot 5 random images from test folder
        print("Selecting 5 images for prediction...")
        select_5_images_for_prediction(predicted_classes, actual_classes, test_dir)
        # Get the confusion matrix
        from sklearn.metrics import confusion_matrix
        print("Calculating confusion matrix...")
        confusion_matrix = confusion_matrix(actual_classes, predicted_classes)
        # Build class index mapping from dataset class names
        class_names = getattr(test_generator, 'class_names', CLASS_NAMES)
        class_index = {name: i for i, name in enumerate(class_names)}
        # Plot the confusion matrix
        plot_confusion_matrix(confusion_matrix, classes=class_index, title="Confusion Matrix")
        # Save the confusion matrix
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        # Calculate metrics
        print("\nCalculating metrics...")
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
            shutil.rmtree(dataset_path)
            # Delete the temporary directory
            shutil.rmtree("C:/tmp")

    if (Evaluation or Prediction) and model is None:
        print("Skipped evaluation/prediction because the model could not be loaded or created.")

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


