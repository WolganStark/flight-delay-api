# Flight Delay API

Python version 3.11.9

## Run locally

```bash
cd flight-delay-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.app:app --reload