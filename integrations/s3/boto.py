import boto3

class BotoS3:
    def __init__(self):
        self.s3_client = self.get_boto_s3_client()

    def get_boto_session(self):
        session = boto3.Session()
        return session

    def get_boto_s3_client(self):
        session = self.get_boto_session()
        s3 = session.client('s3')
        return s3
    


import datetime
import re

def extract_year(text):
    # Use a regular expression to find a 4-digit year in the text
    match = re.search(r'\b\d{4}\b', text)
    if match:
        return str(int(match.group()))
    else:
        return None

def remove_numbers(text):
    # Use a regular expression to remove all digits from the text
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove any leading or trailing whitespace that might result from removing numbers and special characters
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def folder_month_date(month_year_str):
    format_str = '%B %Y'
    datetime_obj = datetime.datetime.strptime(month_year_str, format_str)
    return datetime_obj.strftime('%B').lower()

def folder_month_dt(month_year_str):
    format_str = '%B'
    datetime_obj = datetime.datetime.strptime(month_year_str, format_str)
    return datetime_obj.strftime('%m').lower()


def list_s3_folders(s3_client, bucket_name, prefix):
    folders = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    for page in pages:
        for prefix in page.get('CommonPrefixes', []):
            folders.append(prefix.get('Prefix'))
    return folders

def list_s3_files(s3_client, bucket_name, prefix):
    files = []
    # prefix = prefix.rstrip('/')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    for page in pages:
        for obj in page.get('Contents', []):
            # Ensure the file is directly under the prefix
            key = obj.get('Key')
            if key[len(prefix):].count('/') == 0:  # No additional slashes in the key after the prefix
                files.append(key)
    
    return files

