from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    print(data)
    
    # Extract data from the request
    room_number = int(data['room_number'])
    area = float(data['Area'])
    hasElevator = int(data['hasElevator'])
    hasBars = int(data['hasBars'])
    hasStorage = int(data['hasStorage'])
    hasAirCondition = int(data['hasAirCondition'])
    hasBalcony = int(data['hasBalcony'])
    handicapFriendly = int(data['handicapFriendly'])
    city_אילת = int(data['City_אילת'])
    city_באר_שבע = int(data['City_באר שבע'])
    city_בית_שאן = int(data['City_בית שאן'])
    city_בת_ים = int(data['City_בת ים'])
    city_גבעת_שמואל = int(data['City_גבעת שמואל'])
    city_דימונה = int(data['City_דימונה'])
    city_הוד_השרון = int(data['City_הוד השרון'])
    city_הרצליה = int(data['City_הרצליה'])
    city_זכרון_יעקב = int(data['City_זכרון יעקב'])
    city_חולון = int(data['City_חולון'])
    city_חיפה = int(data['City_חיפה'])
    city_יהוד_מונוסון = int(data['City_יהוד מונוסון'])
    city_ירושלים = int(data['City_ירושלים'])
    city_כפר_סבא = int(data['City_כפר סבא'])
    city_מודיעין_מכבים_רעות = int(data['City_מודיעין מכבים רעות'])
    city_נהרייה = int(data['City_נהרייה'])
    city_נוף_הגליל = int(data['City_נוף הגליל'])
    city_נס_ציונה = int(data['City_נס ציונה'])
    city_נתניה = int(data['City_נתניה'])
    city_פתח_תקווה = int(data['City_פתח תקווה'])
    city_צפת = int(data['City_צפת'])
    city_קרית_ביאליק = int(data['City_קרית ביאליק'])
    city_ראשון_לציון = int(data['City_ראשון לציון'])
    city_רחובות = int(data['City_רחובות'])
    city_רמת_גן = int(data['City_רמת גן'])
    city_רעננה = int(data['City_רעננה'])
    city_שוהם = int(data['City_שוהם'])
    city_תל_אביב = int(data['City_תל אביב'])
    type_אחר = int(data['type_אחר'])
    type_בית_פרטי = int(data['type_בית פרטי'])
    type_בניין = int(data['type_בניין'])
    type_דו_משפחתי = int(data['type_דו משפחתי'])
    type_דופלקס = int(data['type_דופלקס'])
    type_דירה = int(data['type_דירה'])
    type_דירת_גג = int(data['type_דירת גג'])
    type_דירת_גן = int(data['type_דירת גן'])
    type_דירת_נופש = int(data['type_דירת נופש'])
    type_טריפלקס = int(data['type_טריפלקס'])
    type_מגרש = int(data['type_מגרש'])
    type_מיני_פנטהאוז = int(data['type_מיני פנטהאוז'])
    type_פנטהאוז = int(data['type_פנטהאוז'])
    type_קוטג = int(data['type_קוטג'])
    type_קוטג_טורי = int(data['type_קוטג טורי'])
    condition_דורש_שיפוץ = int(data['condition _דורש שיפוץ'])
    condition_חדש = int(data['condition _חדש'])
    condition_ישן = int(data['condition _ישן'])
    condition_לא_צוין = int(data['condition _לא צוין'])
    condition_משופץ = int(data['condition _משופץ'])
    condition_שמור = int(data['condition _שמור'])
    furniture_אין = int(data['furniture _אין'])
    furniture_חלקי = int(data['furniture _חלקי'])
    furniture_לא_צויין = int(data['furniture _לא צויין'])
    furniture_מלא = int(data['furniture _מלא'])
    entrance_date_above_year = int(data['entrance_date_above_year'])
    entrance_date_flexible = int(data['entrance_date_flexible'])
    entrance_date_less_than_6_months = int(data['entrance_date_less_than_6_months'])
    entrance_date_months_6_12 = int(data['entrance_date_months_6_12'])
    entrance_date_not_defined = int(data['entrance_date_not_defined'])
    publishedDays = int(data['publishedDays'])
    floor = int(data['floor'])
    total_floor = int(data['total_floor'])
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'room_number': [room_number],
        'Area': [area],
        'hasElevator': [hasElevator],
        'hasBars': [hasBars],
        'hasStorage': [hasStorage],
        'hasAirCondition': [hasAirCondition],
        'hasBalcony': [hasBalcony],
        'handicapFriendly': [handicapFriendly],
        'City_אילת': [city_אילת],
        'City_באר שבע': [city_באר_שבע],
        'City_בית שאן': [city_בית_שאן],
        'City_בת ים': [city_בת_ים],
        'City_גבעת שמואל': [city_גבעת_שמואל],
        'City_דימונה': [city_דימונה],
        'City_הוד השרון': [city_הוד_השרון],
        'City_הרצליה': [city_הרצליה],
        'City_זכרון יעקב': [city_זכרון_יעקב],
        'City_חולון': [city_חולון],
        'City_חיפה': [city_חיפה],
        'City_יהוד מונוסון': [city_יהוד_מונוסון],
        'City_ירושלים': [city_ירושלים],
        'City_כפר סבא': [city_כפר_סבא],
        'City_מודיעין מכבים רעות': [city_מודיעין_מכבים_רעות],
        'City_נהרייה': [city_נהרייה],
        'City_נוף הגליל': [city_נוף_הגליל],
        'City_נס ציונה': [city_נס_ציונה],
        'City_נתניה': [city_נתניה],
        'City_פתח תקווה': [city_פתח_תקווה],
        'City_צפת': [city_צפת],
        'City_קרית ביאליק': [city_קרית_ביאליק],
        'City_ראשון לציון': [city_ראשון_לציון],
        'City_רחובות': [city_רחובות],
        'City_רמת גן': [city_רמת_גן],
        'City_רעננה': [city_רעננה],
        'City_שוהם': [city_שוהם],
        'City_תל אביב': [city_תל_אביב],
        'type_אחר': [type_אחר],
        'type_בית פרטי': [type_בית_פרטי],
        'type_בניין': [type_בניין],
        'type_דו משפחתי': [type_דו_משפחתי],
        'type_דופלקס': [type_דופלקס],
        'type_דירה': [type_דירה],
        'type_דירת גג': [type_דירת_גג],
        'type_דירת גן': [type_דירת_גן],
        'type_דירת נופש': [type_דירת_נופש],
        'type_טריפלקס': [type_טריפלקס],
        'type_מגרש': [type_מגרש],
        'type_מיני פנטהאוז': [type_מיני_פנטהאוז],
        'type_פנטהאוז': [type_פנטהאוז],
        'type_קוטג': [type_קוטג],
        'type_קוטג טורי': [type_קוטג_טורי],
        'condition _דורש שיפוץ': [condition_דורש_שיפוץ],
        'condition _חדש': [condition_חדש],
        'condition _ישן': [condition_ישן],
        'condition _לא צוין': [condition_לא_צוין],
        'condition _משופץ': [condition_משופץ],
        'condition _שמור': [condition_שמור],
        'furniture _אין': [furniture_אין],
        'furniture _חלקי': [furniture_חלקי],
        'furniture _לא צויין': [furniture_לא_צויין],
        'furniture _מלא': [furniture_מלא],
        'entrance_date_above_year': [entrance_date_above_year],
        'entrance_date_flexible': [entrance_date_flexible],
        'entrance_date_less_than_6_months': [entrance_date_less_than_6_months],
        'entrance_date_months_6_12': [entrance_date_months_6_12],
        'entrance_date_not_defined': [entrance_date_not_defined],
        'publishedDays': [publishedDays],
        'floor': [floor],
        'total_floor': [total_floor]
    })

    # Standardize the input data using the scaler
    predicted_price = model.transform(input_data)[0]

    response = {
        'predicted_price': predicted_price
    }

    return jsonify(response)
    

if __name__ == '__main__':
    app.run(debug=True)
