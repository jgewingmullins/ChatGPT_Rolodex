# Email Rolodex Updater

This script fetches emails from Gmail accounts, processes them, and updates a Google Sheet with the relevant information. It uses OpenAI's API to generate summaries and context for the emails.

## Prerequisites

- Python 3.x
- Gmail accounts with IMAP enabled
- Google Cloud project with Sheets API enabled
- OpenAI API key

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-repo/email-rolodex-updater.git
   cd email-rolodex-updater
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create configuration file**

    Create a config.json file in the root directory with the following structure:
    ```json
    {
        "gmail_accounts": [
            {
                "username": "your-email@gmail.com",
                "password": "your-app-password"
            },
            {
                "username": "your-other-email@gmail.com",
                "password": "your-other-app-password"
            }
        ],
        "days_past": 7,
        "batch_size": 1000,
        "google_sheets": {
            "scope": [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ],
            "creds": "path/to/your/service_account.json",
            "sheet_key": "your-google-sheet-key"
        },
        "openai": {
            "api_key": "your-openai-api-key"
        }
    }
    ```

4. **Run the script**

    By default, the script will ignore internal emails from arcadiascience.com. To run the script with the default settings, use:

   ```bash
   python Rolodex_Script.py
   ```

    If you want to include internal emails, use the --ignore_internal argument:

   ```bash
   python Rolodex_Script.py --ignore_internal False
   ```
