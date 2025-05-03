import os
import shutil
import logging

def sync_directories(source_dir, dest_dir):
    """
    Synchronizes all files from source directory to destination directory,
    creating any necessary directories in the destination.
    
    Args:
        source_dir (str): Path to the source directory
        dest_dir (str): Path to the destination directory
    
    Returns:
        bool: True if sync completed successfully, False otherwise
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger('DirectorySync')
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return False
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
            
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Calculate the relative path
        rel_path = os.path.relpath(root, source_dir)
        
        # Create corresponding directory in destination if it doesn't exist
        if rel_path != '.':
            dest_path = os.path.join(dest_dir, rel_path)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
        else:
            dest_path = dest_dir
        
        # Copy files
        for file in files:
            source_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            
            # Check if file exists in destination and if it's newer in source
            copy_file = True
            if os.path.exists(dest_file):
                src_mtime = os.path.getmtime(source_file)
                dst_mtime = os.path.getmtime(dest_file)
                if src_mtime <= dst_mtime:
                    copy_file = False
            
            if copy_file:
                shutil.copy2(source_file, dest_file)
    
    logger.info("Sync completed successfully")
    return True