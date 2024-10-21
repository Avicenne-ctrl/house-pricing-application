from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import sys
sys.path.append('..')
import script.utilities as ut

data = pd.read_csv("static/Housing 2.csv")

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret_key"

@app.route("/")
@app.route('/', methods=['GET', 'POST'])

def index():
    
    if request.method == 'POST':
        
        area             = request.form.get('area')
        bedrooms         = request.form.get('bedrooms')
        bathrooms        = request.form.get('bathrooms')
        stories          = request.form.get('stories')
        mainroad         = request.form.get('mainroad')
        guestroom        = request.form.get('guestroom')
        basement         = request.form.get('basement')
        hotwaterheating  = request.form.get('hotwaterheating')
        airconditioning  = request.form.get('airconditioning')
        parking          = request.form.get('parking')
        prefarea         = request.form.get('prefarea')
        furnishingstatus = request.form.get('furnishingstatus')
        
        query_dict = {
                    "area": int(area),
                    "bedrooms": int(bedrooms),
                    "bathrooms": int(bathrooms),
                    "stories": int(stories),
                    "mainroad": int(mainroad),
                    "guestroom": int(guestroom),
                    "basement": int(basement),
                    "hotwaterheating": int(hotwaterheating),
                    "airconditioning": int(airconditioning),
                    "parking": int(parking),
                    "prefarea": int(prefarea),
                    "furnishingstatus": int(furnishingstatus),
                }
                
        data_query = ut.create_query_dataframe(query_dict)
        
        x_train, x_val, y_train, y_val= ut.preprocess_data(data)

        xgb = ut.train_and_save_xgboost(x_train, x_val, y_train, y_val)
        
        price = f"{int(ut.make_prediction(data_query, xgb))}$"
        
        return render_template('resultats.html', prediction_value= price)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="3030", threaded=False)


