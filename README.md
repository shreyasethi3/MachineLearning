# AI Chatbot — Dual Mode (Retrieval + Seq2Seq LSTM)

This project implements a dual-mode chatbot:
- Retrieval-based (fast, rule-based)
- Seq2Seq LSTM generator (stronger LSTM; trains to generate responses)

**Train the seq2seq model first** (it can be slow depending on data and hardware):
```bash
python train_tf.py
```

Then run the app:
```bash
python app.py
```
Open http://127.0.0.1:5000 and select the mode (Retrieval or LSTM) and chat.