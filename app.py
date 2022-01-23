# prepare a data set for classification with the decision tree algorithm
from scipy.sparse import data
import pandas as pd
from flask import Flask,jsonify
from flask import request
import numpy as np

# load the model from disk
# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
import pickle
# from waitress import serve


model = pickle.load(open('good.sav','rb'))

app = Flask(__name__)
@app.route('/')
def handle_request():
    return "Hi,this is stress dectection project from Fontys University"
@app.route('/predict',methods=["POST","GET"])
def predict():
    # def onMessage():
    #         pass
    #clean up data
    #preditction = loaded_model.predict(incoming message)
    # send prediction back
    #  mean = request.form.get('MEAN')
    #  max = request.form.get('MAX')
    #  min = request.form.get('MIN')
    #  kurt = request.form.get('KURT')
    #  skew = request.form.get('SKEW')
    parameter1 = request.form['mean']
    parameter2 = request.form['max']
    parameter3 = request.form['min']
    parameter4 = request.form['range']
    parameter5 = request.form['kurt']
    parameter6 = request.form['skew']

    print(parameter1,parameter2,parameter3,parameter4,parameter5,parameter6)

    if parameter1 != 0:
        test2=  [ parameter1, parameter2,parameter3,parameter4,parameter5,parameter6]
        input_query = np.array([test2])
        result = model.predict(input_query)
        if result == [0]:
            result = "Relax"
        else:
            result = "Stress"

        print(result)
        # for i in range(len(result)):
        #     print(result[i])
        return jsonify(result)

    # test1=  [ 3.646182349, 3.837203979, 3.387832642,0.449371338, 0.251447438, -0.815356535]
    
@app.route('/test',methods=["POST","GET"])
def test():
    return "POST TEST DONGD"
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3030,debug=True)


