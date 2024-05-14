# Viral


## Reading from google sheets
* Call the gsheet2df function to the sheet_id (found in the url of the spreadsheet) and the sheet_name 
* You will need to set up credentials so that the function can access the sheet.
* [This link](https://developers.google.com/sheets/api/quickstart/python) has most of the instructions. Repeated below for clarity.
* Click `enable the api`
* Click the three dots on the left, create a new project
* Find the google sheets api
* Activate it
* Go onto the OAuth consent screen (on the left)
* Create a new application
* Click continue though all the steps.
* In the Google Cloud console, go to Menu menu > APIs & Services > Credentials.
* Go to Credentials
* Click Create Credentials > OAuth client ID.
* Click Application type > Desktop app.
* In the Name field, type a name for the credential. This name is only shown in the Google Cloud console.
* Click Create. The OAuth client created screen appears, showing your new Client ID and Client secret.
* Click OK. The newly created credential appears under OAuth 2.0 Client IDs.
* Save the downloaded JSON file as credentials.json in the same directory as this README.
* Go to oauth consent screen and put your app into production from testing


## Multisession analysis
* Update the DATA_PATH in constants.py to the path to the server






