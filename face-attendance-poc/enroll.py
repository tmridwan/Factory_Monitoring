# enroll.py
import cv2
import insightface
import numpy as np
import os
import pickle

# Load the model (same as your test script)
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

KNOWN_FACES_DIR = 'known_faces'
OUTPUT_FILE = 'known_embeddings.pkl'

known_embeddings = {}  # {person_name: [list of embeddings]}

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing {person_name}...")
    embeddings_for_person = []

    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"  Skipping {image_file} - couldn't read image")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"  No face found in {image_file}, skipping")
            continue
        elif len(faces) > 1:
            print(f"  Multiple faces found in {image_file}, using the largest one")

        # If multiple faces detected, take the largest (closest to camera)
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embeddings_for_person.append(face.embedding)
        print(f"  Got embedding from {image_file}")

    if embeddings_for_person:
        # Average all embeddings for this person into one reference vector
        avg_embedding = np.mean(embeddings_for_person, axis=0)
        known_embeddings[person_name] = avg_embedding
        print(f"  Enrolled {person_name} with {len(embeddings_for_person)} photos\n")
    else:
        print(f"  WARNING: No valid faces found for {person_name}, skipping enrollment\n")

# Save to disk
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(known_embeddings, f)

print(f"Enrollment complete. {len(known_embeddings)} people enrolled.")
print(f"Saved to {OUTPUT_FILE}")