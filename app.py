# from flask import Flask, render_template
# import joblib


# app = Flask(__name__)


# # Load ML model
# model_Lg = joblib.load('./regr.pkl')
# model_Dt = joblib.load('./dt.pkl')
                     
# # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'GARAGE']
# pred_Lg = model_Lg.predict([[4, 2.5, 3005, 15,  1]])[0][0].round(1)
# pred_Dt = model_Dt.predict([[4, 2.5, 3005, 15,  1]])[0][0].round(1)

# @app.route('/')
# def index():
#     return render_template('index.html', l_Reg=str(pred_Lg), dt_Reg=str(pred_Dt))


from flask import Flask, render_template
import joblib

app = Flask(__name__)


# Load ML model
model_Lg = joblib.load('./lReg.pkl')
model_Dt = joblib.load('./dtReg.pkl')


# Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'GARAGE']
pred_Lg = model_Lg.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
pred_Dt = model_Dt.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0].round(2)

@app.route('/')
def hello_world():
    # return render_template('index.html', Reg_test=str(pred_test))
    return render_template('index.html', l_Reg=str(pred_Lg), dt_Reg=str(pred_Dt))
