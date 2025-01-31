import pickle
import os.path
import sys
from pathlib import Path
from typing import Dict

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd

from viral.constants import HERE


CRED_PATH = HERE.parent / "credentials.json"
TOKEN_PATH = HERE.parent / "token.pickle"


def build_gsheet(spreadsheet_id: str, sheet_name: str) -> Dict:
    """Takes input of google sheets SPREADSHEET_ID and SHEET_NAME
    returns gsheet object that can be read by gsheet2df into a pandas dataframe.

    The user must follow the instructions here to enable the sheets api in
    their account and download their credentials.json to their working directory.
    (https://developers.google.com/sheets/api/quickstart/python)

    """
    creds = None
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH, SCOPES)
            # JR - this is needed to authenticate through ssh
            # flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    service = build("sheets", "v4", credentials=creds, cache_discovery=False)

    # Call the Sheets API
    sheet = service.spreadsheets()

    # If the sheet name contains numbers, it must be enclosed in single quotes
    if any(char.isdigit() for char in sheet_name):
        sheet_name = f"'{sheet_name}'"

    return sheet.values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()


def gsheet2df(spreadsheet_id: str, sheet_name: str, header_row: int) -> pd.DataFrame:
    """
    Imports the sheet defined in SPREADSHEET_ID as a pandas dataframe
    Inputs -
    spreadsheet_id: found in the spreadsheet URL
                    https://docs.google.com/spreadsheets/d/SPREADSHEET_ID
    header_row:     the row that contains the header - titles of the columns
    sheet_name:     the name of the sheet to import

    Returns -
    df: a pandas dataframe

    Converts gsheet object from build_gsheet to a Pandas DataFrame.
    Use of this function requires the user to follow the instructions
    in build_gsheet.

    Empty cells are represented by ''

    """

    gsheet = build_gsheet(spreadsheet_id, sheet_name)
    header = gsheet.get("values", [])[header_row - 1]
    values = gsheet.get("values", [])[header_row:]

    if not values:
        print("no data found")
        return

    # corrects for rows which end with blank cells
    for i, row in enumerate(values):
        if len(row) < len(header):
            [row.append("") for _ in range(len(header) - len(row))]
            values[i] = row

    all_data = []
    for col_id, col_name in enumerate(header):

        column_data = [row[col_id] for row in values]
        ds = pd.Series(data=column_data, name=col_name)
        all_data.append(ds)

    return pd.concat(all_data, axis=1)
