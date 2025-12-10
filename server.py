# server.py  ← FINAL WORKING VERSION (Dec 2025)
from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import socket
import threading
import time


app = Flask(__name__)

# Load model
try:
    with open("EE_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model EE_model.pkl loaded successfully!")
except Exception as e:
    print("Model load failed:", e)
    exit()

FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

latest_data = {
    "features": {f: 0.0 for f in FEATURES},
    "prediction": 0,
    "probability": 0.5,
    "timestamp": time.time(),
    "status": "Normal",
    "color": "green"
}

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5007))
    print("UDP listener started on port 5007")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            msg = data.decode('utf-8').strip()
            values = [float(x) for x in msg.split(',')]
            if len(values) != 8:
                continue

            # CRITICAL: Use raw numpy array → NO feature names!
            X = np.array(values).reshape(1, -1)

            pred = int(model.predict(X)[0])
            prob = model.predict_proba(X)[0]
            seizure_prob = round(prob[1] if len(prob) > 1 else prob[0], 3)

            latest_data.update({
                "features": {f"F{i+1}": round(v, 3) for i, v in enumerate(values)},
                "prediction": pred,
                "probability": seizure_prob,
                "status": "Seizure Risk!" if pred == 1 else "Normal",
                "color": "red" if pred == 1 else "green",
                "timestamp": time.time()
            })
        except Exception as e:
            print("UDP error (ignored):", e)

threading.Thread(target=udp_listener, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    return jsonify(latest_data)

if __name__ == "__main__":
    ip = socket.gethostbyname(socket.gethostname())
    print("\n" + "="*60)
    print("LIVE DASHBOARD IS READY!")
    print(f"OPEN THIS LINK ON PHONE OR ANY BROWSER:")
    print(f"http://{ip}:5007")
    print("="*60 + "\n")
    
    # CRITICAL: no debug, no reloader → fixes blank page on Windows
    app.run(host="0.0.0.0", port=5007, debug=False, use_reloader=False)