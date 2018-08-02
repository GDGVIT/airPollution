
import urllib.parse
import requests

def find_coordinates(query):

    main_api = 'https://maps.googleapis.com/maps/api/place/textsearch/json?'
    url = main_api + urllib.parse.urlencode({'query': query}) + '&key=YOUR API KEY HERE'

    json_data = requests.get(url).json()
    json_status = json_data['status']
    print('\nAPI Status :' + json_status)

    if json_status == 'OK':

        location_details = {
            'name_of_place': json_data['results'][0]['name'],
            'formatted_address': json_data['results'][0]['formatted_address'],
            'location': json_data['results'][0]['geometry']['location'],
            'upper_left': (json_data['results'][0]['geometry']['viewport']['northeast']['lat'], json_data['results'][0]['geometry']['viewport']['southwest']['lng']),
            'lower_right': (json_data['results'][0]['geometry']['viewport']['southwest']['lat'], json_data['results'][0]['geometry']['viewport']['northeast']['lng']),
                            }
        
        return location_details






