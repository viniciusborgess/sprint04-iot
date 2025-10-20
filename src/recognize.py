#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

import os
import requests

# ================= Guardian integration (config) =================
GUARDIAN_API_URL = os.getenv("GUARDIAN_API_URL", "http://127.0.0.1:8000/api/events")
GUARDIAN_API_TOKEN = os.getenv("GUARDIAN_API_TOKEN", "")

# Overlay state (para mostrar advisory na janela do OpenCV)
last_advisory_text = ""
last_advisory_until = 0.0

# Debounce para não enviar muitos eventos repetidos
_last_sent_ts = 0.0
_last_person = None
_COOLDOWN_SECONDS = float(os.getenv("GUARDIAN_COOLDOWN", "2.5"))


def send_guardian_event(person: str, confidence: float, distance: float):
    """
    Envia um evento para a API do Guardian e captura o advisory de resposta
    para exibir como overlay na janela de vídeo.
    """
    global last_advisory_text, last_advisory_until

    payload = {
        "type": "face_recognized",
        "person": person,
        "confidence": float(confidence),
        "distance": float(distance),
        "timestamp": int(time.time() * 1000),
    }
    headers = {"Content-Type": "application/json"}
    if GUARDIAN_API_TOKEN:
        headers["Authorization"] = f"Bearer {GUARDIAN_API_TOKEN}"

    try:
        r = requests.post(GUARDIAN_API_URL, json=payload, headers=headers, timeout=2.5)
        print("[Guardian] POST", GUARDIAN_API_URL, r.status_code)
        if r.ok:
            try:
                data = r.json()
                adv = data.get("advisory", {})
                msg = adv.get("message") or "Advisory recebido do Guardian."
                # Exibe por 3 segundos
                last_advisory_text = msg[:100]  # limita tamanho
                last_advisory_until = time.time() + 3.0
            except Exception as parse_err:
                print("[Guardian] aviso: não foi possível interpretar JSON:", parse_err)
        else:
            print("[Guardian] erro HTTP:", r.status_code, r.text[:200])
    except Exception as e:
        print("[Guardian] event error:", e)


def draw_label(img, text, x, y):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 10)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 4, y + baseline - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def make_trackbar_window(init_scale=110, init_neighbors=5, init_min_size=60, init_thr=80):
    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 420, 140)
    cv2.createTrackbar("scale x100", "Controles", init_scale, 200, lambda v: None)
    cv2.createTrackbar("neighbors", "Controles", init_neighbors, 15, lambda v: None)
    cv2.createTrackbar("minSize", "Controles", init_min_size, 300, lambda v: None)
    cv2.createTrackbar("lbph_thr", "Controles", init_thr, 200, lambda v: None)


def main():
    global _last_sent_ts, _last_person

    ap = argparse.ArgumentParser(description="Detecção e reconhecimento facial em tempo real")
    ap.add_argument("--model", default="data/model/lbph_model.yaml", help="Caminho do modelo LBPH")
    ap.add_argument("--labels", default="data/model/labels.json", help="Caminho do mapeamento label->nome")
    ap.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    args = ap.parse_args()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if face_cascade.empty():
        raise SystemExit("Falha ao carregar Haar Cascade de face.")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        raise SystemExit("cv2.face não encontrado. Instale 'opencv-contrib-python'.")

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists() or not labels_path.exists():
        raise SystemExit("Modelo ou labels não encontrados. Treine antes com train_lbph.py.")

    recognizer.read(str(model_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a webcam.")

    make_trackbar_window()
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ler sliders
        scale = max(cv2.getTrackbarPos("scale x100", "Controles"), 110) / 100.0
        neigh = max(cv2.getTrackbarPos("neighbors", "Controles"), 1)
        min_size = max(cv2.getTrackbarPos("minSize", "Controles"), 20)
        thr = max(cv2.getTrackbarPos("lbph_thr", "Controles"), 30)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neigh, minSize=(min_size, min_size)
        )

        recognized_person_in_frame = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes[:2]:
                cx, cy = x + ex + ew//2, y + ey + eh//2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)

            face200 = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            try:
                label_id, distance = recognizer.predict(face200)
            except cv2.error:
                label_id, distance = -1, 9999.0

            name = id_to_name.get(str(label_id), "Desconhecido")
            if distance > thr:
                name = "Desconhecido"

            draw_label(frame, f"{name}  dist={distance:.1f}", x, y - 10)

            if name != "Desconhecido" and recognized_person_in_frame is None:
                recognized_person_in_frame = (name, distance)

        # Envia no máximo 1 evento por frame e aplica cooldown
        if recognized_person_in_frame is not None:
            name, distance = recognized_person_in_frame
            now = time.time()
            should_send = False
            if _last_person != name:
                should_send = True
            elif (now - _last_sent_ts) > _COOLDOWN_SECONDS:
                should_send = True

            if should_send:
                # Heurística de confiança: 1 - (distance/200), limitado [0,1]
                conf = float(max(0.0, 1.0 - (distance / 200.0)))
                send_guardian_event(name, confidence=conf, distance=distance)
                _last_person = name
                _last_sent_ts = now

        fps = 1.0 / (time.time() - last) if time.time() != last else 0.0
        last = time.time()
        draw_label(frame, f"scale={scale:.2f} neigh={neigh} minSize={min_size}px thr={thr}", 10, 25)
        draw_label(frame, f"FPS: {fps:.1f}", 10, 50)

        # Overlay do advisory se houver
        if time.time() < last_advisory_until and last_advisory_text:
            draw_label(frame, last_advisory_text, 10, 80)

        cv2.imshow("Reconhecimento - LBPH", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
