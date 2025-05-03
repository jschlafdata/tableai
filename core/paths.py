from pathlib import Path
from typing import Optional, Union
import os
# from core.log import critical, error, warning, info, debug

class PathTranslator:
    """Helper class to handle path translation between absolute, relative, and web paths."""
    
    def __init__(self, absolute_path: Path, base_path: Path = None):
        """
        Initialize a path translator.
        
        Args:
            absolute_path: The absolute path to represent
            base_path: Optional base path for calculating relative paths
        """
        self._absolute = absolute_path
        self._base_path = base_path
        
        # Calculate relative path if base_path is provided
        if base_path:
            try:
                self._relative = absolute_path.relative_to(base_path)
            except ValueError:
                # Fallback if path isn't relative to base_path
                self._relative = absolute_path
        else:
            self._relative = absolute_path

    @property
    def abs(self) -> Path:
        """Return the absolute path."""
        return self._absolute

    @property
    def rel(self) -> Path:
        """Return the path relative to the base path."""
        return self._relative

    @property
    def react(self) -> str:
        """Return a web-friendly path for React components."""
        return f"/outputs/{self._relative}"

    def __str__(self) -> str:
        """String representation defaults to relative path."""
        return str(self._relative)


class Paths:
    """Main path manager to handle project directories."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize path manager with project configuration.
        
        Args:
            input_dir: Path to input directory
            output_dir: Path to output directory
            operator: Name of the operator
            deployment: Deployment identifier
            environment: Environment name (dev, prod, etc.)
        """
        # Resolve the library base path
        self.library_directory = Path(__file__).parent.parent.resolve()
        
        # Store configuration
        self.input_dir = Path(input_dir)
        self.output_dir = Path(self.library_directory) / Path(output_dir)
        self.state_directory = self.output_dir / Path('state')
        self.root_dir = self.library_directory
        self.react_dir = Path(self.library_directory) / Path('ui/pdf-viewer-app/public/outputs')
    
    def get_path(self, path: Union[str, Path], base_path: Path = None) -> PathTranslator:
        """
        Create a PathTranslator for the given path.
        
        Args:
            path: Path to translate (string or Path object)
            base_path: Optional base path for relative calculations
                      (defaults to library directory)
                      
        Returns:
            A PathTranslator object
        """
        # Default to library directory if no base path provided
        if base_path is None:
            base_path = self.library_directory
            
        # Ensure path is absolute
        if isinstance(path, str):
            path = Path(path)
        
        # Make absolute if not already
        if not path.is_absolute():
            absolute_path = (base_path / path).resolve()
        else:
            absolute_path = path.resolve()
            
        return PathTranslator(absolute_path=absolute_path, base_path=base_path)
    
    @property
    def output_path(self) -> PathTranslator:
        """Get a PathTranslator for the output directory."""
        return self.get_path(self.output_dir, self.library_directory)
    
    @property
    def input_path(self) -> PathTranslator:
        """Get a PathTranslator for the input directory."""
        return self.get_path(self.input_dir, self.library_directory)
    
    @property
    def lib_path(self) -> PathTranslator:
        """Get a PathTranslator for the library directory."""
        return self.get_path(self.library_directory, self.library_directory)
    
    @property
    def state_path(self) -> PathTranslator:
        """Get a PathTranslator for the state directory."""
        return self.get_path(self.state_directory, self.library_directory)