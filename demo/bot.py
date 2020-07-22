import pandas as pd
import random
import requests
import sys

bot_responses = [
    {
        "responses": [
            "We open at 8am everyday",
            "We are open from 8am to 10pm everyday",
            "8am to 10pm everyday"
        ],
        "tags": ["hours", 'opening']
    },
    {
        "responses": [
            "Tacos & Burgers",
            "Pizza"
        ],
        "tags": ["menu", 'food']
    }
]



# call REST API locally
# to predict on a query


def predict_and_respond(txt=None, bot_df=None):
    if txt == None and bot_df is None:
        return "Sorry, I don't know what that means. Please contact us."
    json = {
        "query": txt
    }
    r = requests.post("http://127.0.0.1:8000/predict", json=json)
    if r.status_code not in range(200, 299):
        # send a signal, logging
        return "Sorry, I am having trouble right now. Please try again later."
    pred_response = r.json()
    print(pred_response)
    pred_tags = [k for k,v in pred_response['predictions'].items() if v != 0]
    mask = bot_df.tags.apply(lambda x: set(pred_tags) == set(x))
    response_df = bot_df[mask]
    all_responses = list(response_df['responses'].values)
    responses = []
    for row in all_responses:
        for r in row:
            responses.append(r)
    responses = list(set(responses))
    if len(responses) == 0:
        return "Sorry, I am still learning. I don't understand what you said."
    return random.choice(responses)


if __name__ == "__main__":
    bot_df = pd.DataFrame(bot_responses)
    query = "Are you open at 9:30am tomorrow?"
    if len(sys.argv) >= 1:
        query = sys.argv[1]
    response = predict_and_respond(query, bot_df=bot_df)
    print(response)