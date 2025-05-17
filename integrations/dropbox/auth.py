import os
import json
import dropbox
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dotenv import load_dotenv
from pathlib import Path
from api.models.service import ServiceConfig, DropboxAuthState

class DropboxAuth:
    """
    Handles OAuth2 authentication for Dropbox and provides a Dropbox client.
    - Reads APP_KEY and APP_SECRET from env vars DROPBOX_APP_KEY / DROPBOX_APP_SECRET.
    - Caches access (and refresh) tokens in ~/.dropbox_token.json.
    - Provides a simple CLI-based OAuth2 flow if no token is found.
    """
    def __init__(self):
        self.dropbox_state = DropboxAuthState.from_service_config(ServiceConfig())
        if not (self.dropbox_state.app_key and self.dropbox_state.app_secret):
            raise ValueError("You must set DROPBOX_APP_KEY and DROPBOX_APP_SECRET environment variables.")
        self.client = None

    def load_token(self) -> bool:
        """Load saved token from disk (if it exists)."""
        token_file = os.path.expanduser(self.dropbox_state.token_file)
        if os.path.isfile(token_file):
            try:
                with open(token_file, "r") as f:
                    data = json.load(f)
                self.dropbox_state.access_token = data.get("access_token")
                self.dropbox_state.refresh_token = data.get("refresh_token")
                self.dropbox_state.expires_at = data.get("expires_at")
                if not self.dropbox_state.refresh_token:
                    return False
                return True
            except Exception as e:
                print(f"Error loading token: {e}")
                return False
        return False

    def save_token(self, access_token: str, refresh_token: str = None, expires_at: str = None):
        """Save tokens (and optional expiry) to disk."""
        payload = {"access_token": access_token}
        if refresh_token:
            payload["refresh_token"] = refresh_token
        if expires_at:
            payload["expires_at"] = expires_at
        token_file = os.path.expanduser(self.dropbox_state.token_file)
        with open(token_file, "w") as f:
            json.dump(payload, f)

    def authenticate_cli(self):
        """Run the OAuth2 'no redirect' flow in the terminal."""
        flow = DropboxOAuth2FlowNoRedirect(
            self.dropbox_state.app_key,
            self.dropbox_state.app_secret,
            token_access_type="offline"
        )

        auth_url = flow.start()
        print("1. Go to:", auth_url)
        print("2. Click 'Allow' (you may have to log in).")
        code = input("3. Enter the authorization code here: ").strip()

        oauth_result = flow.finish(code)
        self.dropbox_state.access_token = oauth_result.access_token
        self.dropbox_state.refresh_token = getattr(oauth_result, "refresh_token", None)

        self.save_token(self.dropbox_state.access_token, refresh_token=self.dropbox_state.refresh_token)
        print("✔ Authentication successful. Token saved to", self.dropbox_state.token_file)

    def get_client(self) -> dropbox.Dropbox:
        """
        Returns an authenticated dropbox.Dropbox client.
        If no token is loaded, runs the CLI auth flow.
        """
        if not self.dropbox_state.access_token:
            if not self.load_token():
                self.authenticate_cli()

        if self.dropbox_state.refresh_token:
            self.client = dropbox.Dropbox(
                oauth2_access_token=self.dropbox_state.access_token,
                oauth2_refresh_token=self.dropbox_state.refresh_token,
                app_key=self.dropbox_state.app_key,
                app_secret=self.dropbox_state.app_secret,
            )
        else:
            self.client = dropbox.Dropbox(self.dropbox_state.access_token)

        return self.client

if __name__ == "__main__":
    # Example usage
    auth = DropboxAuth()
    dbx = auth.get_client()
    account = dbx.users_get_current_account()
    print("Hello,", account.name.display_name)