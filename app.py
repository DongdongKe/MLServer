# prepare a data set for classification with the decision tree algorithm
from scipy.sparse import data
import pandas as pd
from flask import Flask,request,jsonify
import numpy as np

# load the model from disk
# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
import pickle


model = pickle.load(open('good.sav','rb'))

app = Flask(__name__)
@app.route('/')
def handle_request():
    return "Hi,this is stress dectection project from Fontys University"
@app.route('/predict',methods=['POST'])
def predict():
   #  mean = request.form.get('MEAN')
   #  max = request.form.get('MAX')
   #  min = request.form.get('MIN')
   #  kurt = request.form.get('KURT')
   #  skew = request.form.get('SKEW')
  				
    test1=  [ 3.646182349, 3.837203979, 3.387832642,0.449371338, 0.251447438, -0.815356535]
    test2=  [ 1.646182349, 2.837203979, 1.387832642,0.449371338, -0.951447438, 0.315356535]
    input_query = np.array([test1])
    result = model.predict(input_query)
    if result == [0]:
       result = "Relax"
    else:
         result = "Stress"



    print(result)
    for i in range(len(result)):
        print(result[i])
    return jsonify({'Result':str(result)})
if __name__ == '__main__':
    app.run(port=3030,debug=True)

# def onMessage():
#         pass
#clean up data
#preditction = loaded_model.predict(incoming message)
# send prediction back
