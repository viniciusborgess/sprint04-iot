#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import cv2
import numpy as np

def load_dataset(data_dir: Path):
    images = []
    labels = []
    names = []
    name_to_id = {}

    for person_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        name = person_dir.name
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id)
            names.append(name)
        label_id = name_to_id[name]

        for img_path in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (200, 200):
                img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(label_id)

    return images, np.array(labels, dtype=np.int32), name_to_id

def main():
    ap = argparse.ArgumentParser(description="Treina um modelo LBPH a partir do dataset de faces")
    ap.add_argument("--data-dir", default="data/faces", help="Diretório do dataset")
    ap.add_argument("--model-out", default="data/model/lbph_model.yaml", help="Caminho de saída do modelo")
    ap.add_argument("--labels-out", default="data/model/labels.json", help="Caminho de saída do mapeamento label->nome")
    ap.add_argument("--radius", type=int, default=1, help="Parâmetro radius do LBPH")
    ap.add_argument("--neighbors", type=int, default=8, help="Parâmetro neighbors do LBPH")
    ap.add_argument("--grid-x", type=int, default=8, help="Parâmetro grid_x do LBPH")
    ap.add_argument("--grid-y", type=int, default=8, help="Parâmetro grid_y do LBPH")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit("Dataset não encontrado. Rode primeiro collect_faces.py")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=args.radius, neighbors=args.neighbors, grid_x=args.grid_x, grid_y=args.grid_y
        )
    except AttributeError:
        raise SystemExit("cv2.face não encontrado. Instale 'opencv-contrib-python'.")

    images, labels, name_to_id = load_dataset(data_dir)
    if len(images) == 0:
        raise SystemExit("Nenhuma imagem encontrada. Colete dados antes de treinar.")

    recognizer.train(images, labels)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(model_out))

    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump({str(v): k for k, v in name_to_id.items()}, f, ensure_ascii=False, indent=2)

    # relatório simples
    total = len(images)
    classes = len(name_to_id)
    print(f"[i] Treino concluído - imagens: {total}, pessoas: {classes}")
    print(f"[i] Modelo salvo em: {model_out}")
    print(f"[i] Labels salvos em: {labels_out}")
    print(f"[i] Hiperparâmetros - radius={args.radius}, neighbors={args.neighbors}, grid_x={args.grid_x}, grid_y={args.grid_y}")

if __name__ == "__main__":
    main()
