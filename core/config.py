from data_loaders.ingest_files import FileReader
import os
import json

CONFIG_FILE = "settings.yml"

class CONFIG:
    @staticmethod
    def root(root_dir):
        config_path = os.path.join(root_dir, CONFIG_FILE)
        return FileReader.yaml(config_path)
    
    @staticmethod
    def pretty(config):
        return json.dumps(config, indent=4, sort_keys=True)