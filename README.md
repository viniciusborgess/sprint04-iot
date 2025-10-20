# Sprint 04 – IOT & IOB
## Guardian – Reconhecimento Facial Integrado ao Sistema de Conscientização Financeira

**Integrantes:**
- Vinicius Sobreira Borges – RM 97767
- Leticia Fontana Baptista – RM 550289
- Guilherme Catelli Bichaco – RM 97989
- Julia Palomari – RM 551910
- Julia Ortiz – RM 550204

---

## 🎯 Objetivo
Esta Sprint evolui a POC da **Sprint 03 (reconhecimento facial local)** para uma **integração prática com o Case Guardian**.

O **Guardian** é uma aplicação que busca promover **educação e consciência financeira** através de tecnologia.
Nesta versão, o reconhecimento facial é usado para **identificar o usuário** e **enviar eventos para o sistema Guardian**, que responde com mensagens educativas e personalizadas.

---

## 🧠 Arquitetura da Solução


- O módulo facial detecta e identifica o usuário com **OpenCV + Haar Cascade + LBPH**.
- Quando um rosto é reconhecido, o sistema **dispara um evento REST** para a **API Guardian**.
- A API responde com uma mensagem (*advisory*) personalizada, que aparece na tela do usuário.

---

## ⚙️ Funcionalidades principais
- ✅ Detecção e identificação facial em tempo real
- ✅ Parâmetros ajustáveis de detecção (scale, vizinhos, minSize, threshold)
- ✅ Integração via REST API com o sistema Guardian
- ✅ Exibição de mensagens educativas (advisories) personalizadas
- ✅ Logs detalhados e resposta visual direta


---

## 🧩 Requisitos
- Python 3.9 ou superior
- Webcam funcional
- Sistema operacional: Windows, macOS ou Linux
- Dependências: `opencv-contrib-python`, `fastapi`, `uvicorn`, `requests`, `numpy`

---

## 🚀 Como executar o projeto

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-contrib-python numpy fastapi uvicorn requests


# 1. Rode a API Guardian em um terminal separado
cd guardian_integration
python3 -m uvicorn main:app --reload --port 8000

# 2. Coletar rostos
python3 src/collect_faces.py --name "Vinicius"

# 3. Treinar modelo
python3 src/train_lbph.py

# 4. Rodar reconhecimento facial
python3 src/recognize.py --model data/model/lbph_model.yaml --labels data/model/labels.json --camera 1


# 5. Verifique os eventos em:
http://127.0.0.1:8000/api/events
