
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_data = {}

with open("model.pkl", 'rb') as f:
    model_data = pickle.loads(f.read())

    
class CustomerInput(BaseModel):
    query:str

def predict(txt='Hello world', 
                vectorizer=None, 
                model=None, 
                classes=[], 
                *args, 
                **kwargs):
    # pred
    assert(vectorizer!=None)
    assert(model != None)
    input_vector = vectorizer.transform([txt])
    output_vector = model.predict(input_vector)
    assert(len(output_vector[0]) == len(classes))
    preds = {}
    for i, val in enumerate(output_vector[0]):
        preds[classes[i]] = int(val)
    return preds

@app.post("/predict")
def predict_view(customer_input:CustomerInput):
    # storing this query data -> SQL database
    my_pred = predict(customer_input.query, **model_data)
    return {"query": customer_input.query, "predictions": my_pred}

# @app.post('/train')
