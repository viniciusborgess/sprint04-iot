#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
import time

def draw_label(img, text, x, y):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 10)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 4, y + baseline - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Coleta imagens de rosto para dataset")
    ap.add_argument("--name", required=True, help="Nome da pessoa a ser coletada")
    ap.add_argument("--out", default="data/faces", help="Pasta de saída do dataset")
    ap.add_argument("--num", type=int, default=40, help="Quantidade de imagens a capturar")
    ap.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    ap.add_argument("--scale", type=float, default=1.10, help="scaleFactor do Haar Cascade")
    ap.add_argument("--neighbors", type=int, default=5, help="minNeighbors do Haar Cascade")
    ap.add_argument("--min-size", type=int, default=60, help="minSize em pixels do Haar Cascade")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    if face_cascade.empty():
        raise RuntimeError("Falha ao carregar Haar Cascade de face. Verifique instalação do OpenCV.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    saved = 0
    last = time.time()

    print("[i] Pressione 'c' ou 'space' para salvar uma captura. 'q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=(args.min_size, args.min_size)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            # pequenos "landmarks" com detector de olhos
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.10, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes[:2]:
                cx, cy = x + ex + ew//2, y + ey + eh//2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)

        fps = 1.0 / (time.time() - last) if time.time() != last else 0.0
        last = time.time()

        draw_label(frame, f"salvos: {saved}/{args.num}", 10, 25)
        draw_label(frame, f"scale={args.scale:.2f} neigh={args.neighbors} minSize={args.min_size}px", 10, 50)
        draw_label(frame, f"FPS: {fps:.1f}", 10, 75)

        cv2.imshow("Coleta de rostos", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('c'), ord(' ')]:
            # salvar a maior face encontrada
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
                filename = out_dir / f"{args.name}_{saved:03d}.png"
                cv2.imwrite(str(filename), roi)
                saved += 1
                print(f"[+] Salvo {filename}")
                if saved >= args.num:
                    print("[i] Coleta finalizada.")
                    break
            else:
                print("[!] Nenhum rosto detectado para salvar.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
