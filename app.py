from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

activities_df = pd.read_csv('activities.csv')
# Load the model and encoder
with open('activities.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('activities_encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/attraction')
def Attraction():
    return render_template('attraction.html')

@app.route('/activities', methods=['GET', 'POST'])
def Activities():
    recommendations = []
    if request.method == 'POST':
        location_input = request.form['location']
        type_input = request.form['adventure_type']

        # Create input DataFrame and encode
        input_data = pd.DataFrame({'type': [type_input], 'location': [location_input]})
        input_encoded = encoder.transform(input_data)
        input_encoded = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['type', 'location']))

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        recommendations = get_top_names(type_input, location_input, prediction)

    return render_template('activities.html', recommendations=recommendations)

@app.route('/dining')
def Dining():
    return render_template('dining.html')

def get_top_names(type_input, location_input, prediction):
    # Filter and sort the activities based on type and location
    filtered_activities = activities_df[
        (activities_df['type'] == type_input) & (activities_df['location'] == location_input)
    ]
    sorted_activities = filtered_activities.sort_values(by='ratings', ascending=False)

    # Get the top 3 activities
    top_names = sorted_activities['name'].head(3).tolist()
    return top_names

if __name__ == '__main__':
    app.run(debug=True, port=5000)
