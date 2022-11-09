import requests
import json

class epi_calc:

        def __init__(self, key, series_id, start_year, end_year):

            headers = {'Content-type': 'application/json'}
            parameters = json.dumps({'seriesid' : series_id, 'startyear' : start_year, 'endyear' : end_year, 'registrationkey' : key})

            self.request(headers, parameters)

        def request(self, headers, parameters):
            
            post = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data = parameters, headers = headers)
            json_data = json.loads(post.text)

            print(json_data)

key = '99512ee9e46544ea81ff8b37ec590d3f' #registration key for Robert as registered through BLS
series_id = 'CUUR0000SA0' #series id for CPI for All Urban Consumers (CPI-U): All items in U.S. city average, all urban consumers, not seasonally adjusted
start_year = 2000
end_year = 1980

epi = epi_calc(key, series_id, start_year, end_year)

#ABOVE CODE IS NOT WORKING, ABANDONING API FOR NOW. SWITCHING TO MANUAL DOWNLOAD

