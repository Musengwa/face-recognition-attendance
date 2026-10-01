# open_CV — Face Recognition Attendance Engine

This folder contains the Python/OpenCV prototype for the recognition engine described in the [project README](../README.md): it watches a webcam feed, recognizes known faces, and logs each recognized name to a text file as a simple attendance record.

## How it works

### 1. `face_encoding.py` — the `Face` class

- **`load_encoding_images(images_path)`** — scans a folder of reference photos (one clear face per image, filename = person's name, e.g. `Yogesh.jpg`). For each image it computes a 128-value face encoding (via the `face_recognition` library) and stores it alongside the person's name (taken from the filename). This is the "known faces" database, built once at startup.
- **`detect_known_faces(frame)`** — takes a single video frame:
  1. Shrinks it to 25% size (`frame_resizing = 0.25`) so detection runs faster.
  2. Converts it from OpenCV's BGR color format to the RGB format `face_recognition` expects.
  3. Finds all face locations in the frame and computes an encoding for each detected face.
  4. Compares each detected encoding against every known encoding and picks the closest match (smallest face distance). If the closest match is within the library's default tolerance, it's labeled with that person's name; otherwise it's labeled `"Unknown"`.
  5. Scales the face coordinates back up to the original frame size (since detection ran on the shrunk frame) and returns `(face_locations, face_names)`.

### 2. `Face Detection Based Attendance System.py` — the main script

This is the entry point you run:

1. Loads all reference images from a `Dataset/` folder (create this folder and add one photo per person, named after them).
2. Opens `attendance_log.txt` in append mode and starts the default webcam (`cv2.VideoCapture(0)`).
3. In a loop, reads each frame, runs `detect_known_faces` on it, then for every recognized face:
   - Draws a rectangle and the person's name on the video window.
   - The first time a name is seen in this session, writes it to `attendance_log.txt` (a `unique_names` set prevents duplicate log lines within the same run).
4. Press **`q`** to stop the loop, close the window, and release the camera.

`attendance_log.txt` currently just accumulates a plain list of names that showed up (see the sample entries already in the file) — there's no timestamp or date yet.

### 3. `concept_study/`

Standalone learning scripts used while picking up OpenCV basics — not part of the attendance pipeline:
- `imageProcessing1.py` / `displayImage.py` — basic image loading/display experiments (`myImage.png`, `output.jpg`).
- `videoProcessing1.py` — basic video capture experiments.

## Setup

1. **Install dependencies**:
   ```bash
   pip install opencv-python face_recognition numpy
   ```
   `face_recognition` depends on `dlib`, which needs CMake and a C++ build toolchain to compile on Windows. If `pip install face_recognition` fails, install a prebuilt `dlib` wheel first, or install CMake and Visual Studio Build Tools, then retry.

2. **Add reference photos**: create a `Dataset/` folder inside `open_CV/` and add one image per known person, named after them (e.g. `Dataset/Yogesh.jpg`). Each image should contain exactly one clearly visible face.

3. **Run it**:
   ```bash
   python "Face Detection Based Attendance System.py"
   ```
   A window will open showing the webcam feed with recognized names boxed. Press `q` to quit.

## Known limitations

- Attendance is only deduplicated per run (in-memory `unique_names` set) — restarting the script lets the same name be logged again.
- No date/timestamp is recorded with each name.
- Uses the default webcam index (`0`); change this in the script if you have multiple cameras.
- This is the standalone prototype; it isn't yet wired up to the `blockchain` or `frontend` folders described in the top-level project README.
