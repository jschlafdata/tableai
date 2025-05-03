import os
import json
import dropbox
from dropbox.oauth import DropboxOAuth2FlowNoRedirect

# Where to store your token (adjust path if desired):
TOKEN_FILE = os.path.expanduser("~/.dropbox_token.json")

class DropboxAuth:
    """
    Handles OAuth2 authentication for Dropbox and provides a Dropbox client.

    - Reads APP_KEY and APP_SECRET from env vars DROPBOX_APP_KEY / DROPBOX_APP_SECRET.
    - Caches access (and refresh) tokens in ~/.dropbox_token.json.
    - Provides a simple CLI-based OAuth2 flow if no token is found.
    """

    def __init__(self, app_key: str = None, app_secret: str = None, token_file: str = TOKEN_FILE):
        self.app_key = app_key or os.getenv("DROPBOX_APP_KEY")
        self.app_secret = app_secret or os.getenv("DROPBOX_APP_SECRET")
        if not (self.app_key and self.app_secret):
            raise ValueError("You must set DROPBOX_APP_KEY and DROPBOX_APP_SECRET environment variables.")
        self.token_file = token_file
        self.access_token = None
        self.refresh_token = None
        self.client = None

    def load_token(self) -> bool:
        """Load saved token from disk (if it exists)."""
        if os.path.isfile(self.token_file):
            with open(self.token_file, "r") as f:
                data = json.load(f)
            self.access_token = data.get("access_token")
            self.refresh_token = data.get("refresh_token")
            if not self.refresh_token:
                return False
            return True
        return False

    def save_token(self, access_token: str, refresh_token: str = None, expires_at: str = None):
        """Save tokens (and optional expiry) to disk."""
        payload = {"access_token": access_token}
        if refresh_token:
            payload["refresh_token"] = refresh_token
        if expires_at:
            payload["expires_at"] = expires_at
        with open(self.token_file, "w") as f:
            json.dump(payload, f)

    def authenticate_cli(self):
        """
        Run the OAuth2 "no redirect" flow in the terminal:
          1. Prints a URL to visit
          2. User pastes back the auth code
          3. Exchanges it for a token and saves it
        """
        # Key change: request offline access so Dropbox issues a refresh token
        flow = DropboxOAuth2FlowNoRedirect(
            self.app_key,
            self.app_secret,
            token_access_type="offline"
        )

        auth_url = flow.start()
        print("1. Go to:", auth_url)
        print("2. Click 'Allow' (you may have to log in).")
        code = input("3. Enter the authorization code here: ").strip()

        oauth_result = flow.finish(code)
        self.access_token = oauth_result.access_token
        # Refresh token only present if offline access is enabled
        self.refresh_token = getattr(oauth_result, "refresh_token", None)

        self.save_token(self.access_token, refresh_token=self.refresh_token)
        print("✔ Authentication successful. Token saved to", self.token_file)

    def get_client(self) -> dropbox.Dropbox:
        """
        Returns an authenticated dropbox.Dropbox client.
        If no token is loaded, runs the CLI auth flow.
        """
        if not self.access_token:
            if not self.load_token():
                self.authenticate_cli()

        # Build client; include refresh_token if available (allows auto-refresh)
        if self.refresh_token:
            self.client = dropbox.Dropbox(
                oauth2_access_token=self.access_token,
                oauth2_refresh_token=self.refresh_token,
                app_key=self.app_key,
                app_secret=self.app_secret,
            )
        else:
            self.client = dropbox.Dropbox(self.access_token)

        return self.client


if __name__ == "__main__":
    # Example usage
    auth = DropboxAuth()
    dbx = auth.get_client()
    account = dbx.users_get_current_account()
    print("Hello,", account.name.display_name)