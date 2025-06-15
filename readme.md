## Face Recognition System (Flask + Webcam UI)

A simple face recognition API built using Flask, OpenCV, and `face_recognition`. It allows:

- Face registration from webcam
- Real-time face recognition via webcam
- Storing face data locally
- RESTful API for management

---

### 🔧 Features

- Register a face using webcam (saves encoding + photo)
- Real-time face recognition via webcam stream
- Delete specific registered faces
- List all registered faces
- Base64 image transfer between frontend and backend
- Cross-platform and runs locally

---

### 📁 Project Structure

```
face-recognition-system/
│
├── backend/
│   ├── app.py                 # Flask backend
│   ├── known_faces/           # Stores images and encodings.pkl
│
├── frontend/
│   ├── register.html          # UI for face registration
│   ├── recognize.html         # UI for live recognition
│
└── README.md
```

---

### ✅ Requirements

- Python 3.8+
- `pip install -r requirements.txt` (see below)

---

### 📦 Install Dependencies

```bash
pip install flask flask-cors opencv-python face-recognition numpy
```

> Note: You may need `dlib` installed with CMake if using Linux.

---

### 🚀 Running the Flask Backend

```bash
cd backend
python app.py
```

By default, it runs at:
**[http://localhost:5000](http://localhost:5000)**

---

### 🖥️ Run the Frontend (HTML UI)

Just open the HTML files in your browser:

- `frontend/register.html` – for capturing and registering faces
- `frontend/recognize.html` – for real-time face recognition

Make sure the backend is running before using them.

---

### 🧠 API Endpoints

| Method | Endpoint              | Description                  |
| ------ | --------------------- | ---------------------------- |
| POST   | `/register`           | Register a new face          |
| POST   | `/recognize`          | Recognize face from image    |
| GET    | `/registered_faces`   | Get list of registered names |
| DELETE | `/delete_face/<name>` | Delete a registered face     |

---

### 📝 Example API Call (Register)

```bash
curl -X POST http://localhost:5000/register \
-H "Content-Type: application/json" \
-d '{"name": "John", "image": "data:image/jpeg;base64,/9j/4AAQSkZ..."}'
```

---

### 🛑 Known Limitations

- Face encodings are stored in memory & `pickle` — not secure for production
- No database or authentication (for simplicity)
- One face registration at a time
- Works best under good lighting

---

### 📸 Preview

**Register UI**

- Opens webcam
- Capture + submit face

**Recognize UI**

- Runs recognition every 2 seconds
- Shows matched names with confidence

---

### 📌 To Do

- [ ] Add bounding boxes in UI
- [ ] Store encodings in SQLite/PostgreSQL
- [ ] Add JWT authentication
- [ ] Dockerize for deployment

---

### 👨‍💻 Author

Built by [Prathamesh](https://github.com/prathamesh-b) – Software Engineer & Maker

---
