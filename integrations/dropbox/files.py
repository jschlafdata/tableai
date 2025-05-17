from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
import datetime
from dropbox.files import FileMetadata, FolderMetadata, ExportInfo, FolderSharingInfo


@dataclass
class DropboxMetadataConverter:
    """Dataclass to convert Dropbox FileMetadata objects to dictionaries"""
    
    @staticmethod
    def _convert_datetime(dt: Optional[datetime.datetime]) -> Optional[str]:
        """Convert datetime objects to ISO format strings"""
        return dt.isoformat() if dt is not None else None
    
    @staticmethod
    def _convert_export_info(export_info: Optional[ExportInfo]) -> Optional[Dict[str, Any]]:
        """Convert ExportInfo objects to dictionaries"""
        if export_info is None:
            return None
        
        return {
            "export_as": export_info.export_as,
            "export_options": export_info.export_options
        }
    
    @staticmethod
    def _convert_sharing_info(share_info: Optional[FolderSharingInfo]) -> Optional[Dict[str, Any]]:
        """Convert ExportInfo objects to dictionaries"""
        if share_info is None:
            return None
        
        result={}
        for attr in ['no_access', 'parent_shared_folder_id', 'read_only', 'shared_folder_id', 'traverse_only']:
            if hasattr(share_info, attr):
                value = getattr(share_info, attr)
            else:
                value = None
            result[attr] = value
        return result
    
    @staticmethod
    def to_dict(metadata: Union[FileMetadata, FolderMetadata]) -> Dict[str, Any]:
        """Convert a Dropbox metadata object to a dictionary"""
        result = {}

        result['dropbox_id'] = getattr(metadata, 'id')
        # Common attributes for both file and folder metadata
        for attr in ['name', 'path_display', 'path_lower', 'parent_shared_folder_id', 
                    'sharing_info', 'property_groups', 'has_explicit_shared_members']:
            if hasattr(metadata, attr):
                value = getattr(metadata, attr)
                # Skip attributes with value None or NOT_SET
                if value is not None and value != 'NOT_SET':
                    result[attr] = value

        if isinstance(metadata, FolderMetadata):
            result['type']='folder'
            if hasattr(metadata, 'sharing_info') and metadata.sharing_info is not None:
                result['sharing_info'] = DropboxMetadataConverter._convert_sharing_info(metadata.sharing_info)
        # FileMetadata specific attributes
        if isinstance(metadata, FileMetadata):
            result['type']='file'
            # Handle datetime attributes
            if hasattr(metadata, 'client_modified'):
                result['client_modified'] = DropboxMetadataConverter._convert_datetime(metadata.client_modified)
                
            if hasattr(metadata, 'server_modified'):
                result['server_modified'] = DropboxMetadataConverter._convert_datetime(metadata.server_modified)
            
            # Handle export_info
            if hasattr(metadata, 'export_info') and metadata.export_info is not None:
                result['export_info'] = DropboxMetadataConverter._convert_export_info(metadata.export_info)

            if hasattr(metadata, 'sharing_info') and metadata.sharing_info is not None:
                result['sharing_info'] = DropboxMetadataConverter._convert_sharing_info(metadata.sharing_info)
            
            # Add other FileMetadata attributes
            for attr in ['content_hash', 'is_downloadable', 'size', 'media_info', 
                         'symlink_info', 'file_lock_info', 'preview_url', 'rev']:
                if hasattr(metadata, attr):
                    value = getattr(metadata, attr)
                    if value is not None and value != 'NOT_SET':
                        result[attr] = value
        return result