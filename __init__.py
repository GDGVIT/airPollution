from flask import Flask, jsonify, request
from main import location_based_estimation, coordinates_based_estimation

app = Flask(__name__)
 
@app.route("/location/<place>")
def for_particular_place(place):
    return jsonify(location_based_estimation(place))

@app.route("/coordinates/")
def for_particular_coordinates():
    ullat, ullon = float(request.args.get('ullat')), float(request.args.get('ullon'))
    lrlat, lrlon = float(request.args.get('lrlat')), float(request.args.get('lrlon'))
    return jsonify(coordinates_based_estimation(ullat, ullon, lrlat, lrlon))

if __name__ == '__main__':
   app.run(debug = True)