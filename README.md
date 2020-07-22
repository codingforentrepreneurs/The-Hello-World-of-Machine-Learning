[![The Hello World of Machine Learning Logo](https://static.codingforentrepreneurs.com/media/cfe-blog/the-hello-world-of-machine-learning/The_Hello_World_of_Machine_Learning_-_Post.jpg)](https://www.codingforentrepreneurs.com/blog/the-hello-world-of-machine-learning/)


Machine learning is simply a computer learning from data instead of following a recipe. It's meant to mimic how people (and perhaps other animals) learn while still being grounded in mathematics.

This post is meant to get you started with a basic machine learning model. 

A chatbot.

Now, we're not re-creating Alexa, Siri, Cortana, or Google Assistant but we are going to create a brand new machine learning program from scratch. 

This tutorial is meant to be easy assuming you know a bit of Python Programming.

Watch the [entire series](/projects/hello-world-machine-learning) that corresponds to this post.

### Step 1: What's our data?

Machine learning needs data to actually, well, learn. Machines don't yet learn like you and I do but they do learn by finding patterns in things that may seem non-obvious to you and I. We'll see a lot of that in this entire post.

Before we define our data, let's talk about the goal of this ML (machine learning) project:
>  To answer somewhat "random" questions with pre-defined responses.


Here's what we'll try and solve:

__Scenario 1__

Bill: `Hi there, what time do you open tomorrow for lunch?`

Bot: `Our hours are 9am-10pm everyday.`

__Scenario 2__

Karen: `Can I speak to your manager?`

Bot: `You can contact our customer support at 555-555-555.5`


__Scenario 3__

Wade: `What type of products do you have?`

Bot: `We carry various food items including tacos, nachos, burritos, and salads.`

Let's put this into a python format:


```python
conversations = [
    {
        "customer": "Hi there, what time do you open tomorrow for lunch?",
        "response": "Our hours are 9am-10pm everday."
    },
     {
        "customer": "Can I speak to your manager?",
        "response": "You can contact our customer support at 555-555-5555."
    },
     {
        "customer": "What type of products do you have?",
        "response": "We carry various food items including tacos, nachos, burritos, and salads."
    }  
    
]
```

Without machine learning our bot would look like this (uncomment next cell to run):


```python
# while True:
#     my_input = input("What is your question?\n")
#     response = None
#     for convo in conversations:
#         if convo['customer'] == my_input:
#             response = convo['response']
#     if response != None:
#         print(response)
#         break
#     print("I don't know")
#     continue
```

Right away, you should see the huge flaws in this recipe; if a customer doesn't ask a question in a specific pre-defined way, the bot fails and ultimately really sucks. 

A few examples:
    - What if a customer says, _when do you open?_ What do you already know the response to be? 
    - What if a customer says, _Do you sell burgers?_
    - What if a customer says, _How do I reach you on the phone?_
   
I'm sure you could come up with many many more examples of where this really falls apart.

So let's clean up our conversations data a bit more by adding `tags` that describe the initial question.


```python
convos_one = [
    {
        "customer": "Hi there, what time do you open tomorrow for lunch?",
        "tags": ["opening", "closing", "hours"],
    },
     {
        "customer": "Can I speak to your manager?",
        "tags": ["customer_support"],
    },
    {
        "customer": "The food was amazing thank you!",
        "tags": ["customer_support", "feedback"],
    },
     {
      "customer": "What type of products do you have?",
       "tags": ["products", "menu", "inventory", "food"],
    }  
    
]
```


```python
convos_two = [
    {
        "customer": "How late is your kitchen open?",
        "tags": ["opening", "hours", "closing"],
    },
     {
        "customer": "My order was prepared incorrectly, how can I get this fixed?",
        "tags": ["customer_support"],
    },
    {
        "customer": "What kind of meats do you have?",
        "tags": ["menu", "products", "inventory", "food"],
    }
]
```


```python
convos_three = [
    {
        "customer": "When does your dining room open?",
        "tags": ['opening', 'hours'],
    },
     {
        "customer": "When do you open for dinner?",
        "tags": ['opening', 'hours', "closing"],
    },
    {
        "customer": "How do I contact you?",
        "tags": ["contact", "customer_support"]
    }
]
```

Do you see a trend happening here? It's really easy to come up with all kinds of questions for a restaurant bot. It's also easy to see how challenging this would be to try and hard-code conditions to handle all the kinds of queries/questions customers could have.

I'm sure you've heard you need a LOT of data for machine learning. I'll just add one thing to that, you need a lot of data to have *awe-inspiring* machine learning projects. A simple bot for a mom-and-pop store down the street doesn't need *awe-inspiring* just yet. They need simple, approachable, easy to explain. That's exactly what this is. It's not a black box of *millions* of lines of data points. It's like 20 questions with made up on the spot tags.

In so many ways, machine learning today (in the 2020s) is like the internet of the 1990s. People have heard about it and "sort of get it" and feel like it's just this magical gimmick that only super nerds know how to do. Ha. Super nerds.

Now that we have our starting data, let's prepare for machine learning.

First, let's combine all conversations:


```python
dataset = convos_one + convos_two + convos_three
dataset
```




    [{'customer': 'Hi there, what time do you open tomorrow for lunch?',
      'tags': ['opening', 'closing', 'hours']},
     {'customer': 'Can I speak to your manager?', 'tags': ['customer_support']},
     {'customer': 'The food was amazing thank you!',
      'tags': ['customer_support', 'feedback']},
     {'customer': 'What type of products do you have?',
      'tags': ['products', 'menu', 'inventory', 'food']},
     {'customer': 'How late is your kitchen open?',
      'tags': ['opening', 'hours', 'closing']},
     {'customer': 'My order was prepared incorrectly, how can I get this fixed?',
      'tags': ['customer_support']},
     {'customer': 'What kind of meats do you have?',
      'tags': ['menu', 'products', 'inventory', 'food']},
     {'customer': 'When does your dining room open?',
      'tags': ['opening', 'hours']},
     {'customer': 'When do you open for dinner?',
      'tags': ['opening', 'hours', 'closing']},
     {'customer': 'How do I contact you?',
      'tags': ['contact', 'customer_support']}]



Our conversations have the keys `customer` and `tags`. These are arbitrary names for this project and you change change them at-will. Just remember that `customer` equals `input` and `tags` equals `output`. This makes sense because in the future, we want a random customer input such as `What's the menu specials today` and a predicted tags output like `menu` or something similar.


Machine learning has all kinds of terms and acronyms that often make it a bit confusing. In general, just remember that you have some `inputs` and some target `outputs`. Here's what I mean by that:

- `customer`: These values are really the `input` values for our ML project. Input values are sometimes called `source`, `feature`, `training`, `X`, `X_train`/`X_test`/`X_valid`, and a few others.
- `tags`: These values are really the `output` values for our ML project. Output values are sometimes called `target`, `labels`, `y`, `y_train`/`y_test`/`y_valid`, `classes`/`class`, and a few others.

> We're using a machine learning technique known as `supervised learning` which means we provide both the `inputs` and `outputs` to the model. Both data points are known data that we came up with. As you know, the `tags` (or `labels`/`outputs`) have been decided by a human (ie you and me) but can, eventually, be decided by a ML model itself and then verified by a human. Doing so would make the model better and better. There are many other techniques but `supervised learning` is by far the most approachable for beginners.


### Prepare for ML

Now that we have our data, it's time to put it into a format that works well for computers. As you may know, computers are great at numbers and not so great at text. In this case, we have to convert our text into numbers.


This is made simple by using the [scikit-learn](https://scikit-learn.org/stable/index.html) library. So let's install it below by uncommenting the cell.


```python
# !pip install scikit-learn
```

First up, let's turn our `customer` and `tag` data into 2 separate lists where the index of each item corresponds to the index of the other.

```
X = [customer_convo_1, customer_convo_2, ...]
y = [convo_1_tags, convo_2_tags, ...]
```

This is very standard practice so that `X[0]` is the `input` that corresponds to the `y[0]` `output`, `X[1]` is the `input` that corresponds to the `y[1]` `output` and so on. 


```python
inputs = [x['customer'] for x in dataset]
print(inputs)
```

    ['Hi there, what time do you open tomorrow for lunch?', 'Can I speak to your manager?', 'The food was amazing thank you!', 'What type of products do you have?', 'How late is your kitchen open?', 'My order was prepared incorrectly, how can I get this fixed?', 'What kind of meats do you have?', 'When does your dining room open?', 'When do you open for dinner?', 'How do I contact you?']



```python
outputs = [x['tags'] for x in dataset]
print(outputs)
```

    [['opening', 'closing', 'hours'], ['customer_support'], ['customer_support', 'feedback'], ['products', 'menu', 'inventory', 'food'], ['opening', 'hours', 'closing'], ['customer_support'], ['menu', 'products', 'inventory', 'food'], ['opening', 'hours'], ['opening', 'hours', 'closing'], ['contact', 'customer_support']]



```python
assert(len(inputs) == len(outputs))
```

> If you have an `AssertionError` above, that means your `inputs` and `outputs` are not balanced. Check your data source(s) to ensure every `input` has a corresponding `output` value.

Let's verify the positions of this data to show how little we actually changed the data:


```python
idx = 4
print(inputs[idx], outputs[idx])
print(dataset[idx])
```

    How late is your kitchen open? ['opening', 'hours', 'closing']
    {'customer': 'How late is your kitchen open?', 'tags': ['opening', 'hours', 'closing']}


#### The Prediction Function

The goal of machine learning is to produce a function that takes `inputs` and produces `outputs` (predictions). Below is, at a conceptual level, a representation of that:


```python
def my_pred_function(inputs):
    # pred
    outputs = inputs * 0.39013 # this decimal represents what our model will essentially do.
    return outputs
```

Now we need to turn each `inputs` list and `outputs` list into matrices so our machine learning can do machine learning. 

`scikit-learn` has a simple way to do this. First, let's focus on the `inputs` (aka `customer` conversations) as they are the most simple.


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(inputs)
```


> Technical note: `scikit-learn` converted our data into a collection of 1 dimension matrices. We need to use matrices so we can do matrix multiplication (that's how machine learning works under the hood). In `numpy` speak, `X` is an `array` of `array`s.  If you want to see the actual vectors created, check out `X.toarray()` and you'll see it.


```python
X.shape
```




    (10, 43)



`X.shape` is useful to describe our data. 

`X.shape[0]` refers to the number of conversations from our `final_convos` list. So, `X.shape[0] == len(final_convos)` and `X.shape[0] == len(inputs)`


`X.shape[1]` refers to the number of `words` our data has. The `CountVectorizer` did this for us. The Machine Learning term is `features` related to what our data has. You can see all of the `features` (`words` minus punctuation) with:


```python
words = vectorizer.get_feature_names()
print(words)
```

    ['amazing', 'can', 'contact', 'dining', 'dinner', 'do', 'does', 'fixed', 'food', 'for', 'get', 'have', 'hi', 'how', 'incorrectly', 'is', 'kind', 'kitchen', 'late', 'lunch', 'manager', 'meats', 'my', 'of', 'open', 'order', 'prepared', 'products', 'room', 'speak', 'thank', 'the', 'there', 'this', 'time', 'to', 'tomorrow', 'type', 'was', 'what', 'when', 'you', 'your']


The vectorizer has a very limited vocabulary as you can see. Naturally, this means our ML project will *always* misunderstand some key conversations and that's okay. The goal for our project is to get it working first, get customers (or ourselves) using it so we can *improve* it with new data right away (and thus re-improve it).


```python
len(words)
```




    43



#### Prepare Outputs (`labels`)

Every one of our inputs has a list of tags, not just one tag. Let's look at what I mean:



```python
print(inputs[0], outputs[0], isinstance(outputs[0], list))
```

    Hi there, what time do you open tomorrow for lunch? ['opening', 'closing', 'hours'] True


In machine learning, this means `multi-label` classification because there are multiple possible `output` values for each `input` value. This is a more challenging problem than a `single` label but definitely necessary for a chatbot project.

A single label dataset would look like the following:
```
Input: Hi there, how are you doing today?
Output: not_spam

Input: Free CELL phones just text 3ED#2
Output: spam
```

Notice that the output is a single `str` and not a `list` of `str` values. If we continued down this path, our data would *always* fall into 2 categories: `spam` or `not_spam`. This type of classification is called `binary` classification because there are only 2 possible classes for the prediction to be. 


In our project, our `input` values *can* fall into multiple `class` items, 1 `class`, or no `class` item at all. (Remember, `class` = `output tag` = `label`)


```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(outputs)
```

You might consider running `fit_transform` on a `CountVectorizer` like we did with our training data (aka `inputs`) but that doesn't work on multi-label classification. For that, we need `MultiLabelBinarizer`.


```python
mlb.classes_
```




    array(['closing', 'contact', 'customer_support', 'feedback', 'food',
           'hours', 'inventory', 'menu', 'opening', 'products'], dtype=object)



Calling `mlb.classes_` gives us the exact order of how our classes are defined in `y`. So `y[0]` corresponds to `outputs[0]` but in numbers instead of words. It's pretty cool. To see this technically, run the following code:

```
print(y[0])
# map to classes with `zip`
y0_mapped_to_classes = dict(zip(mlb.classes_, y[0]))
print(y0_mapped_to_classes)
```

Then compare:
```
sorted(outputs[0]) == sorted([k for k,v in y0_mapped_to_classes.items() if v == 1])
```



```python
y
```




    array([[1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
           [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
           [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])



Here we can see the matrix that is generated for us with sklearn's `MultiLabelBinarizer`. It's an array of one-hot arrays. 

> **one-hot** is a term that refers to the type of encoding we're using for this particular model. It's a very common practice in machine learning. Essentially turning data into `1`s and `0`s instead of strings or any other data type.


```python
y.shape
```




    (10, 10)



`y.shape` is useful to describe our data in a similar way to `X.shape`

`y.shape[0]` refers to the number of conversations from our `final_convos` list. So, `y.shape[0] == len(final_convos)` and `y.shape[0] == len(outputs)` and `y.shape[0] == X.shape[0]`


`y.shape[1]` refers to the unique values of all of the possible `tags` each conversation has; it will never repeat using the `MultiLabelBinarizer`.


```python
assert y.shape[0] == X.shape[0]
assert y.shape[0] == len(inputs)
```

If you see an `AssertionError` here, it's the same exact error as `assert len(inputs) == len(outputs)` from above. Your data is not balanced.

### Training with `scikit-lean`


```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=1)
model = MultiOutputClassifier(forest, n_jobs=-1)
```


```python
model.fit(X, y)
```




    MultiOutputClassifier(estimator=RandomForestClassifier(random_state=1),
                          n_jobs=-1)



### Prediction


```python
txt = "Hi, when do you close?"
input_vector = vectorizer.transform([txt])
input_vector
```




    <1x43 sparse matrix of type '<class 'numpy.int64'>'
    	with 4 stored elements in Compressed Sparse Row format>




```python
output_vector = model.predict(input_vector)
print(output_vector)
```

    [[0 0 0 0 0 0 0 0 0 0]]



```python
preds = {}
classes = mlb.classes_
for i, val in enumerate(output_vector[0]):
    preds[classes[i]] = val
```


```python
preds
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 0,
     'inventory': 0,
     'menu': 0,
     'opening': 0,
     'products': 0}




```python
def label_predictor(txt='Hello world'):
    # pred
    input_vector = vectorizer.transform([txt])
    output_vector = model.predict(input_vector)
    preds = {}
    classes = mlb.classes_
    for i, val in enumerate(output_vector[0]):
        preds[classes[i]] = val
    return preds
```


```python
label_predictor()
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 0,
     'inventory': 0,
     'menu': 0,
     'opening': 0,
     'products': 0}




```python
label_predictor("When do you open?")
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 1,
     'inventory': 0,
     'menu': 0,
     'opening': 1,
     'products': 0}




```python
label_predictor("When are you opening tomorrow?")
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 0,
     'inventory': 0,
     'menu': 0,
     'opening': 0,
     'products': 0}



### Export Model for Re-Use


```python
import pickle
# classes
# model
# vectorizer

model_data = {
    "classes": list(mlb.classes_),
    "model": model,
    "vectorizer": vectorizer
}

with open("model.pkl", 'wb') as f:
    pickle.dump(model_data, f)
```

### Re-use Exported Model


```python
model_loaded_data = {}

with open("model.pkl", 'rb') as f:
    model_loaded_data = pickle.loads(f.read())

def label_predictor_from_export(txt='Hello world', 
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
    classes = mlb.classes_
    for i, val in enumerate(output_vector[0]):
        preds[classes[i]] = val
    return preds

label_predictor_from_export("When does your kitchen close?", **model_loaded_data)
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 1,
     'inventory': 0,
     'menu': 0,
     'opening': 1,
     'products': 0}



### Retraining with New Data


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def train(dataset, train_col='customer', label_col='tags', export_path='model.pkl'):
    inputs = [x[train_col] for x in dataset]
    outputs = [x[label_col] for x in dataset]
    assert(len(inputs) == len(outputs))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(inputs)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(outputs)
    classes = list(mlb.classes_)
    forest = RandomForestClassifier(random_state=1)
    model = MultiOutputClassifier(forest, n_jobs=-1)
    model.fit(X, y)
    model_data = {
        "classes": list(mlb.classes_),
        "model": model,
        "vectorizer": vectorizer
    }
    with open(export_path, 'wb') as f:
        pickle.dump(model_data, f)
    return export_path
```


```python
# dataset
```


```python
train(dataset, export_path='model2.pkl')
```




    'model2.pkl'




```python
model_loaded_data = {}

with open("model2.pkl", 'rb') as f:
    model_loaded_data = pickle.loads(f.read())
    
label_predictor_from_export("What is your favorite menu item?", **model_loaded_data)
```




    {'closing': 0,
     'contact': 0,
     'customer_support': 0,
     'feedback': 0,
     'food': 0,
     'hours': 0,
     'inventory': 0,
     'menu': 0,
     'opening': 0,
     'products': 0}



### Store Dataset with Pandas


```python
!pip install pandas
```

    Requirement already satisfied: pandas in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (1.0.5)
    Requirement already satisfied: numpy>=1.13.3 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from pandas) (1.19.1)
    Requirement already satisfied: pytz>=2017.2 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from pandas) (2020.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: six>=1.5 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from python-dateutil>=2.6.1->pandas) (1.15.0)



```python
import pandas as pd
```


```python
df = pd.DataFrame(dataset)
df.head(n=100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hi there, what time do you open tomorrow for l...</td>
      <td>[opening, closing, hours]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Can I speak to your manager?</td>
      <td>[customer_support]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The food was amazing thank you!</td>
      <td>[customer_support, feedback]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What type of products do you have?</td>
      <td>[products, menu, inventory, food]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How late is your kitchen open?</td>
      <td>[opening, hours, closing]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>My order was prepared incorrectly, how can I g...</td>
      <td>[customer_support]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What kind of meats do you have?</td>
      <td>[menu, products, inventory, food]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>When does your dining room open?</td>
      <td>[opening, hours]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>When do you open for dinner?</td>
      <td>[opening, hours, closing]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>How do I contact you?</td>
      <td>[contact, customer_support]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_pickle("dataset.pkl")
```


```python
# df = pd.read_pickle("dataset.pkl")
# og_df.head()
```


```python
# og_df.iloc[0]['tags'][0]
```


```python
new_dataset = df.to_dict("records")
# print(new_dataset)
```

### Adding to the Dataset


```python
# df = df.append({"customer": "Who is the manager?", "tags": ["customer_support"]}, ignore_index=True)
```


```python
# df.head(n=100)
```


```python
def append_to_df(df):
    df = df.copy()
    while True:
        customer_input = input("What is the question?\n")
        tags_input = input("Tags? Use commas to separate\n")
        if tags_input != None:
            tags_input = tags_input.split(",")
            if not isinstance(tags_input, list):
                tags_input = [tags_input]
        if customer_input != None and tags_input != None:
            df = df.append({"customer": customer_input, "tags": tags_input}, ignore_index=True)
        tag_another = input("Tag another? Type (y) to continue or any other key to exit.")
        if tag_another.lower() == "y":
            continue
        break
    return df
```


```python
new_df = append_to_df(df)
```

    What is the question?
    Who is the manager?
    Tags? Use commas to separate
    customer_support
    Tag another? Type (y) to continue or any other key to exit.d



```python
new_df.head(n=100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hi there, what time do you open tomorrow for l...</td>
      <td>[opening, closing, hours]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Can I speak to your manager?</td>
      <td>[customer_support]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The food was amazing thank you!</td>
      <td>[customer_support, feedback]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What type of products do you have?</td>
      <td>[products, menu, inventory, food]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How late is your kitchen open?</td>
      <td>[opening, hours, closing]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>My order was prepared incorrectly, how can I g...</td>
      <td>[customer_support]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What kind of meats do you have?</td>
      <td>[menu, products, inventory, food]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>When does your dining room open?</td>
      <td>[opening, hours]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>When do you open for dinner?</td>
      <td>[opening, hours, closing]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>How do I contact you?</td>
      <td>[contact, customer_support]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Who is the manager?</td>
      <td>[customer_support]</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df.to_pickle("dataset.pkl")
```

### Creating a Rest API Model Service
Using [fastapi](https://fastapi.tiangolo.com/)


```python
!pip install fastapi uvicorn requests
```

    Requirement already satisfied: fastapi in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (0.60.0)
    Requirement already satisfied: uvicorn in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (0.11.6)
    Requirement already satisfied: requests in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (2.24.0)
    Requirement already satisfied: pydantic<2.0.0,>=0.32.2 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from fastapi) (1.6.1)
    Requirement already satisfied: starlette==0.13.4 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from fastapi) (0.13.4)
    Requirement already satisfied: websockets==8.* in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from uvicorn) (8.1)
    Requirement already satisfied: uvloop>=0.14.0; sys_platform != "win32" and sys_platform != "cygwin" and platform_python_implementation != "PyPy" in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from uvicorn) (0.14.0)
    Requirement already satisfied: h11<0.10,>=0.8 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from uvicorn) (0.9.0)
    Requirement already satisfied: httptools==0.1.*; sys_platform != "win32" and sys_platform != "cygwin" and platform_python_implementation != "PyPy" in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from uvicorn) (0.1.1)
    Requirement already satisfied: click==7.* in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from uvicorn) (7.1.2)
    Requirement already satisfied: chardet<4,>=3.0.2 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from requests) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from requests) (1.25.9)
    Requirement already satisfied: idna<3,>=2.5 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from requests) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/cfe/.local/share/virtualenvs/ml-hello-world-rt2goiiz/lib/python3.8/site-packages (from requests) (2020.6.20)



```python
API_APP_PATH = 'app.py' # pathlib, os.path
```


```python
# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def homepage_view():
#     return {"Hello": "World"}
```


```python
%%writefile $API_APP_PATH

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
```

    Overwriting app.py



```python
import requests

json = {
    "query": "When do you open?"
}

r = requests.post("http://127.0.0.1:8000/predict", json=json)
print(r.json())
```

    {'query': 'When do you open?', 'predictions': {'closing': 0, 'contact': 0, 'customer_support': 0, 'feedback': 0, 'food': 0, 'hours': 1, 'inventory': 0, 'menu': 0, 'opening': 1, 'products': 0}}



```python
# label_predictor_from_export("When does your kitchen close?", **model_loaded_data)
```

### Responses from Predictions


```python
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

bot_df = pd.DataFrame(bot_responses)
bot_df.head(n=100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>responses</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[We open at 8am everyday, We are open from 8am...</td>
      <td>[hours, opening]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Tacos &amp; Burgers, Pizza]</td>
      <td>[menu, food]</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred_response = {'query': 'When do you open?', 'predictions': {'closing': 0, 'contact': 0, 'customer_support': 0, 'feedback': 0, 'food': 0, 'hours': 1, 'inventory': 0, 'menu': 0, 'opening': 1, 'products': 0}}
```


```python
pred_tags = [k for k,v in pred_response['predictions'].items() if v != 0]
pred_tags
```




    ['hours', 'opening']




```python
mask = bot_df.tags.apply(lambda x: set(pred_tags) == set(x))
print(mask)
```

    0     True
    1    False
    Name: tags, dtype: bool



```python
response_df = bot_df[mask] # bot_df[bot_df.tags.isin("abcs")]
response_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>responses</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[We open at 8am everyday, We are open from 8am...</td>
      <td>[hours, opening]</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_responses = list(response_df['responses'].values)
print(all_responses)
```

    [['We open at 8am everyday', 'We are open from 8am to 10pm everyday', '8am to 10pm everyday']]



```python
responses = []
for row in all_responses:
    for r in row:
        responses.append(r)

responses = list(set(responses))

responses
```




    ['We are open from 8am to 10pm everyday',
     'We open at 8am everyday',
     '8am to 10pm everyday']




```python
import random

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
```


```python
predict_and_respond("Are you open at 9:30am tomorrow?", bot_df=bot_df)
```




    "Sorry, I am still learning. I don't understand what you said."



### What's Next?

1. Get more data, a lot more
- Add more data right now. Keep adding data.
- Refine the data, move tags, remove tags, upgrade.
2. Deploy to a production server:
- Ideally using [this project](https://www.codingforentrepreneurs.com/projects/serverless-container-python-app) for deploying a serverless application using FastAPI (like we did).
3. Use internally, a lot.
- If this tool becomes the go-to for finding answers about your business internally, it can become the same tool for external (customer facing) as well. When in doubt, give it to customers,


```python

```
