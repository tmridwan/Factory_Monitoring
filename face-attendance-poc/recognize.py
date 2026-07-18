# recognize.py
import cv2
import insightface
import numpy as np
import pickle
import time

# Load the model
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(320, 320))

# Load enrolled faces
with open('known_embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)

known_names = list(known_embeddings.keys())
known_vectors = np.array([known_embeddings[name] for name in known_names])

print(f"Loaded {len(known_names)} known people: {known_names}")

# Matching threshold — tune this based on testing
THRESHOLD = 0.5

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recognize_face(embedding):
    best_match = "Unknown"
    best_score = -1

    for name, known_vec in zip(known_names, known_vectors):
        score = cosine_similarity(embedding, known_vec)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_match, best_score

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Process every Nth frame to save compute (not every single frame needs full detection)
FRAME_SKIP = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        t0 = time.time()
        faces = app.get(frame)
        print(f"Detection took {time.time() - t0:.3f}s, found {len(faces)} faces")

        for face in faces:
            bbox = face.bbox.astype(int)
            name, score = recognize_face(face.embedding)

            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()