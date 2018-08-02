from flask import Flask, jsonify
app = Flask(__name__)
from main import project_air_pollution

@app.route('/location/<place>', methods=['POST'])
def process_page(place):
    return place
 
@app.route("/api/<place>")
def json_message(place):

    return jsonify(project_air_pollution(place))

    # return jsonify(message="Hello World from {}".format(place))

if __name__ == '__main__':
   app.run(debug = True)