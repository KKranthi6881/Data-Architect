# SQL-Analytics
# SQL-Analytics
cd knowledge-chat-backend
virtual env create: python3 -m venv venv
activate vm: source venv/bin/activate
pip3 install -r requirements.txt

start FastAPI:  uvicorn src.app:app --reload

Streamlit on : python -m streamlit run src/streamlit_app.py