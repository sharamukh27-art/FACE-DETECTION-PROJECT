# Webcam Emotion Detection on macOS

- `webcam_emotion_test.py`
- `trained_emotion_model.keras`
- `requirements-webcam.txt`

If you want them to test another saved model, send that `.keras` file too.

## Setup

Open Terminal in the folder containing the files.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-webcam.txt
```

## Run

```bash
python webcam_emotion_test.py
```

Press `q` to close the webcam window.

If the default camera does not open, try:

```bash
python webcam_emotion_test.py --camera 1
```

To use a different model file:

```bash
python webcam_emotion_test.py --model second_model.keras
```

## macOS Camera Permission

If the camera window opens but no webcam feed appears, allow Terminal or the code editor to use the camera:

`System Settings` -> `Privacy & Security` -> `Camera`

Then close Terminal and run the script again.
