import pandas as pd
import imaplib
import email
from email.utils import parseaddr, parsedate_to_datetime
import datetime
import re
from unidecode import unidecode
from collections import defaultdict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import openai
import json

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Gmail credentials and server setup
GMAIL_ACCOUNTS = config["gmail_accounts"]
DAYS_PAST = config["days_past"]
BATCH_SIZE = config["batch_size"]

# Google Sheets setup
scope = config["google_sheets"]["scope"]
creds = ServiceAccountCredentials.from_json_keyfile_name(config["google_sheets"]["creds"], scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(config["google_sheets"]["sheet_key"]).sheet1

# OpenAI setup
api_key = config["openai"]["api_key"]
openai.api_key = api_key

# Define maximum email body length for CSV
MAX_EMAIL_BODY_LENGTH = 500  # You can adjust this value as needed

def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and part.get('Content-Disposition') is None:
                body = part.get_payload(decode=True)
                break
    else:
        body = msg.get_payload(decode=True)
    if isinstance(body, bytes):
        body = body.decode('utf-8', errors='ignore')
    body = unidecode(body)  # Remove weird characters
    body = re.sub(r'\s+', ' ', body).strip()  # Remove excessive whitespace
    return body[:MAX_EMAIL_BODY_LENGTH]  # Truncate email body if too long

def has_ics_attachment(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/calendar':
                return True
    return False

def format_email(email_address):
    start = email_address.find('<') + 1
    end = email_address.find('>', start)
    return email_address[start:end] if start > 0 and end > start else email_address


def fetch_emails(mail, folder="inbox", days_past=DAYS_PAST, batch_size=BATCH_SIZE):
    mail.select(folder)
    date_N_days_ago = (datetime.datetime.now() - datetime.timedelta(days=days_past)).strftime("%d-%b-%Y")
    result, data = mail.search(None, f'(SINCE "{date_N_days_ago}")')
    if result != 'OK':
        print("No messages found!")
        return defaultdict(list), defaultdict(list)

    mail_ids = data[0].split()
    email_texts = defaultdict(list)
    email_details = defaultdict(list)

    for batch_start in range(0, len(mail_ids), batch_size):
        batch_ids = mail_ids[batch_start:batch_start + batch_size]
        batch_id_str = ','.join(batch_id.decode('utf-8') for batch_id in batch_ids)
        result, data = mail.fetch(batch_id_str, '(RFC822)')
        if result != 'OK':
            continue

        for response_part in data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                if has_ics_attachment(msg):
                    continue
                email_address = format_email(parseaddr(msg['from'])[1]) if folder == "INBOX" else format_email(parseaddr(msg['to'])[1])
                if 'arcadiascience' not in email_address:
                    body = get_email_body(msg)
                    if 'out of the office' not in body.lower() or 'out of office' not in body.lower():
                        email_texts[email_address].append((body, parsedate_to_datetime(msg['date'])))
                        email_details[email_address].append(parsedate_to_datetime(msg['date']))

    return email_texts, email_details

def fetch_sent_emails(mail, days_past=DAYS_PAST, batch_size=BATCH_SIZE):
    mail.select('"[Gmail]/Sent Mail"')
    date_N_days_ago = (datetime.datetime.now() - datetime.timedelta(days=days_past)).strftime("%d-%b-%Y")
    result, data = mail.search(None, f'(SINCE "{date_N_days_ago}")')
    if result != 'OK':
        print("No messages found!")
        return defaultdict(list), defaultdict(list)

    mail_ids = data[0].split()
    sent_texts = defaultdict(list)
    sent_dates = defaultdict(list)

    for batch_start in range(0, len(mail_ids), batch_size):
        batch_ids = mail_ids[batch_start:batch_start + batch_size]
        batch_id_str = ','.join(batch_id.decode('utf-8') for batch_id in batch_ids)
        result, data = mail.fetch(batch_id_str, '(RFC822)')
        if result != 'OK':
            continue

        for response_part in data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                email_address = format_email(parseaddr(msg['to'])[1])
                date_header = msg['date']
                body = get_email_body(msg)
                if date_header:
                    datetime_sent = parsedate_to_datetime(date_header)
                    sent_dates[email_address].append(datetime_sent)
                    sent_texts[email_address].append((body, datetime_sent))

    return sent_texts, sent_dates

def get_sender_name(email_address):
    name, address = parseaddr(email_address)
    return name if name else address.split('@')[0]


def process_account(username, password):
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(username, password)
    
    # Fetch sent emails to get the most recent contact dates and filter recipients
    email_texts_sent, email_details_sent = fetch_sent_emails(mail, DAYS_PAST)
    filtered_sent_texts = {key: email_texts_sent[key] for key in email_texts_sent if len(email_details_sent[key]) >= 2}
    sent_dates = fetch_sent_emails(mail, DAYS_PAST)[1]

    # Fetch received emails
    email_texts_inbox, email_details_inbox = fetch_emails(mail, "INBOX", DAYS_PAST)

    # Combine texts and details from INBOX for recipients who meet the criteria
    combined_texts = defaultdict(list)
    combined_details = defaultdict(list)
    last_email_bodies_from_me = defaultdict(list)

    for key in filtered_sent_texts:
        if key in email_texts_inbox:
            combined_texts[key].extend(email_texts_inbox[key])
            combined_details[key].extend(email_details_inbox[key])
        # Find the most recent email bodies sent by me
        if key in email_texts_sent:
            last_email_bodies_from_me[key] = sorted(email_texts_sent[key], key=lambda x: x[1], reverse=True)[:3]

    # Process data for output
    data = []
    for email, texts in combined_texts.items():
        texts.sort(key=lambda x: x[1])  # Sort by date
        first_email = texts[0][0]
        longest_email = max(texts, key=lambda x: len(x[0]))[0]

        third_most_recent_email = texts[-3][0] if len(texts) >= 3 else ""
        second_most_recent_email = texts[-2][0] if len(texts) >= 2 else ""
        most_recent_email = texts[-1][0] if len(texts) >= 1 else ""

        last_email_bodies_from_me[email] = last_email_bodies_from_me.get(email, ["", "", ""])
        third_last_email_from_me = last_email_bodies_from_me[email][2][0] if len(last_email_bodies_from_me[email]) >= 3 else ""
        second_last_email_from_me = last_email_bodies_from_me[email][1][0] if len(last_email_bodies_from_me[email]) >= 2 else ""
        last_email_from_me = last_email_bodies_from_me[email][0][0] if len(last_email_bodies_from_me[email]) >= 1 else ""

        # Remove duplicates
        if longest_email == first_email:
            longest_email = ""
        if most_recent_email in {first_email, longest_email}:
            most_recent_email = ""

        last_contacted = max(sent_dates[email]).strftime('%Y-%m-%d') if sent_dates[email] else 'N/A'  # Format as date only
        data.append((get_sender_name(email), email, first_email, longest_email, third_most_recent_email, second_most_recent_email, most_recent_email, third_last_email_from_me, second_last_email_from_me, last_email_from_me, last_contacted))

    return data

all_data = []
for account in GMAIL_ACCOUNTS:
    account_data = process_account(account['username'], account['password'])
    all_data.extend(account_data)

# Load existing data from Google Sheets
existing_data = sheet.get_all_records()

# Convert existing data to DataFrame
existing_df = pd.DataFrame(existing_data)

# Convert new data to DataFrame
new_df = pd.DataFrame(all_data, columns=["Name", "Email Address", "First Email Body", "Longest Email Body", "Third Most Recent Email Body", "Second Most Recent Email Body", "Most Recent Email Body", "Third Last Email Body From Me", "Second Last Email Body From Me", "Last Email Body From Me", "Date Last Contacted"])

# Merge new data with existing data
merged_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["Email Address"], keep='last')

# Sort DataFrame by 'Date Last Contacted' in descending order
merged_df['Date Last Contacted'] = pd.to_datetime(merged_df['Date Last Contacted'], errors='coerce')
merged_df = merged_df.sort_values(by='Date Last Contacted', ascending=False)
merged_df['Date Last Contacted'] = merged_df['Date Last Contacted'].dt.strftime('%Y-%m-%d')

# Function to calculate the number of tokens
def calculate_tokens(text):
    if isinstance(text, str):
        return len(text) // 4
    return 0

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'>+.*\n', '', text)  # Remove lines starting with ">"
        text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
    return text

def generate_summary(name, first_email, longest_email, last_email):
    messages = [
        {"role": "system", "content": "You are an assistant that summarizes emails."},
        {"role": "user", "content": (
            f"Generate a brief summary for the following contact:\n\n"
            f"Name: {name}\n\n"
            f"First Email: {first_email}\n\n"
            f"Longest Email: {longest_email}\n\n"
            f"Most Recent Email: {last_email}\n\n"
            f"The summary should include who the person is, our relationship, any meetings we've had, our shared goals, and any other relevant information. "
            f"Keep the summary to no more than 3 sentences."
        )}
    ]
    
    response = GPT_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=270,  
        n=1,
        temperature=0.7
    )
    
    summary = response.choices[0].message.content.strip()
    summary = re.sub(r'^Summary for \w+:|^Summary:', '', summary).strip()  # Remove "Summary:" or "Summary for [name]:"
    
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return summary, input_tokens, output_tokens

def generate_where_we_left_off(name, email1, email2, email3):
    messages = [
        {"role": "system", "content": "You are an assistant that summarizes email conversations."},
        {"role": "user", "content": (
            f"Summarize the following email conversation for contact {name}:\n\n"
            f"Email 1: {email1}\n\n"
            f"Email 2: {email2}\n\n"
            f"Email 3: {email3}\n\n"
            f"The summary should capture the key points of the conversation and indicate where the discussion was left off. Keep the summary concise."
        )}
    ]
    
    response = GPT_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,  
        n=1,
        temperature=0.7
    )
    
    summary = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return summary, input_tokens, output_tokens

# Initialize token usage counters
CUMULATIVE_INPUT_TOKEN_LIMIT = 500000
CUMULATIVE_OUTPUT_TOKEN_LIMIT = 30000
INPUT_COST_PER_1M_TOKENS = 0.50  # $0.50 per 1M tokens for input
OUTPUT_COST_PER_1M_TOKENS = 1.50  # $1.50 per 1M tokens for output

cumulative_input_tokens_used = 0
cumulative_output_tokens_used = 0
total_cost = 0

# Clean email bodies and apply the cleaning function to the email body columns
merged_df['First Email Body'] = merged_df['First Email Body'].apply(clean_text)
merged_df['Longest Email Body'] = merged_df['Longest Email Body'].apply(clean_text)
merged_df['Most Recent Email Body'] = merged_df['Most Recent Email Body'].apply(clean_text)
merged_df['Second Most Recent Email Body'] = merged_df['Second Most Recent Email Body'].apply(clean_text)
merged_df['Third Most Recent Email Body'] = merged_df['Third Most Recent Email Body'].apply(clean_text)
merged_df['Last Email Body From Me'] = merged_df['Last Email Body From Me'].apply(clean_text)
merged_df['Second Last Email Body From Me'] = merged_df['Second Last Email Body From Me'].apply(clean_text)
merged_df['Third Last Email Body From Me'] = merged_df['Third Last Email Body From Me'].apply(clean_text)

# Generate the summary and "Where we left off" columns
summaries = []
where_we_left_off = []
i = 0

while i < len(merged_df):
    row = merged_df.iloc[i]
    name = row['Name']
    first_email = row['First Email Body'] if pd.notna(row['First Email Body']) else ""
    longest_email = row['Longest Email Body'] if pd.notna(row['Longest Email Body']) else ""
    most_recent_email = row['Most Recent Email Body'] if pd.notna(row['Most Recent Email Body']) else ""
    
    email1 = row['Third Most Recent Email Body'] if pd.notna(row['Third Most Recent Email Body']) else ""
    email2 = row['Second Most Recent Email Body'] if pd.notna(row['Second Most Recent Email Body']) else ""
    email3 = row['Most Recent Email Body'] if pd.notna(row['Most Recent Email Body']) else ""
    email4 = row['Third Last Email Body From Me'] if pd.notna(row['Third Last Email Body From Me']) else ""
    email5 = row['Second Last Email Body From Me'] if pd.notna(row['Second Last Email Body From Me']) else ""
    email6 = row['Last Email Body From Me'] if pd.notna(row['Last Email Body From Me']) else ""

    # Calculate tokens for this round
    tokens_for_summary = calculate_tokens(first_email) + calculate_tokens(longest_email) + calculate_tokens(most_recent_email)
    tokens_for_left_off = calculate_tokens(email1) + calculate_tokens(email2) + calculate_tokens(email3 + " " + email4 + " " + email5 + " " + email6)
    
    if (cumulative_input_tokens_used + tokens_for_summary + tokens_for_left_off > CUMULATIVE_INPUT_TOKEN_LIMIT or
        cumulative_output_tokens_used + 285 > CUMULATIVE_OUTPUT_TOKEN_LIMIT):  # Approximate output tokens per summary
        summaries.append("Summary generation skipped due to token limit.")
        where_we_left_off.append("Summary generation skipped due to token limit.")
        i += 1
        continue

    summary, prompt_tokens, completion_tokens = generate_summary(name, first_email, longest_email, most_recent_email)
    where_we_left_off_summary, prompt_tokens2, completion_tokens2 = generate_where_we_left_off(name, email1, email2, email3 + " " + email4 + " " + email5 + " " + email6)

    # Update tokens used
    cumulative_input_tokens_used += prompt_tokens + prompt_tokens2
    cumulative_output_tokens_used += completion_tokens + completion_tokens2

    # Calculate costs
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    input_cost2 = (prompt_tokens2 / 1_000_000) * INPUT_COST_PER_1M_TOKENS
    output_cost2 = (completion_tokens2 / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
    total_cost += input_cost + output_cost + input_cost2 + output_cost2

    # Print cost information
    print(f"Model: gpt-3.5-turbo")
    print(f"Input tokens used: {prompt_tokens + prompt_tokens2}, Output tokens used: {completion_tokens + completion_tokens2}")
    print(f"Input cost: ${input_cost + input_cost2:.5f}, Output cost: ${output_cost + output_cost2:.5f}")
    print(f"Total cost so far: ${total_cost:.5f}\n")

    summaries.append(summary)
    where_we_left_off.append(where_we_left_off_summary)
    i += 1
    
    # Pause for 2 seconds between API calls
    time.sleep(2)

# Add the summaries to the DataFrame
merged_df['Summary'] = summaries + [""] * (len(merged_df) - len(summaries))  # Fill remaining rows with empty summaries
merged_df['Where we left off'] = where_we_left_off + [""] * (len(merged_df) - len(where_we_left_off))  # Fill remaining rows with empty summaries

# Select relevant columns for the new DataFrame
output_df = merged_df[['Name', 'Email Address', 'Date Last Contacted', 'Summary', 'Where we left off']]

def extract_name_from_summary(summary):
    messages = [
        {"role": "system", "content": "You are an assistant that summarizes emails."},
        {"role": "user", "content": (
            f"Give me the full name of the person who is described in this summary. I only want their name, don't give me any preamble. \n\n{summary}"
        )}
    ]
    
    response = GPT_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=270,  
        n=1,
        temperature=0.7
    )

    name = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return name, input_tokens, output_tokens


# Apply function to extract name from summary
total_name_extraction_cost = 0
for index, row in output_df.iterrows():
    if row['Summary']:
        name, name_input_tokens, name_output_tokens = extract_name_from_summary(row['Summary'])
        output_df.at[index, 'Name'] = name
        
        # Calculate cost for name extraction
        name_input_cost = (name_input_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
        name_output_cost = (name_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
        total_name_extraction_cost += name_input_cost + name_output_cost

# Print total name extraction cost
print(f"Total cost for name extraction: ${total_name_extraction_cost:.5f}")
print(f"Total cost including name extraction: ${total_cost + total_name_extraction_cost:.5f}")

# Convert DataFrame to list of lists for Google Sheets
output_data = output_df.values.tolist()

# Update Google Sheets
sheet.clear()
sheet.append_row(output_df.columns.tolist())
sheet.append_rows(output_data)

print("Rolodex successfully updated in Google Sheets.")
