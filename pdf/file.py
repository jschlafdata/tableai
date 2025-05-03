import os
from dataclasses import dataclass, field, asdict
from typing import List, Union
from core.time import TimeModel  # import your TimeModel here
import uuid
from pathlib import Path
import fitz
import json
from core.serialize import safe_json
from core.paths import PathTranslator
from typing import Dict
from pdf.pdf import Pdf

@dataclass(kw_only=True)
class DirectoryFileNode(TimeModel):
    """
    Represents a single file discovered by traversing a directory tree.
    Inherits timestamp handling from TimeModel (created_at, updated_at, last_sync).
    
    Attributes:
        base_dir (str): The top-level directory from which traversal began.
        sub_dir (str): The path (relative to base_dir) where the file is located.
        file_name (str): The name of the file.
        file_type (str): The file's extension (e.g., 'txt' for a .txt file).
    """
    input_dir: Path
    output_dir: Path
    input_path: PathTranslator
    output_path: PathTranslator
    sub_dir: str
    file_name: str
    full_path: str
    file_type: str
    name: str
    uuid: str
    stage_paths: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    pdfs: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization hook to ensure the time fields from TimeModel are
        properly set. Then add any additional initialization or validation logic.
        """
        super().__post_init__()
        # You can place any additional initialization logic here, if needed.

    def store_metadata(self, stage_number, data: Dict):
        stage = f"stage{stage_number}"
        self.metadata[stage] = {**self.stage_paths[stage_number], **data}
    
    def load_pdf(self, stage_number):
        stage = f"stage{stage_number}"
        abs_path = self.stage_paths[stage_number].get('abs_path', None)
        _pdf = Pdf(abs_path, stage)
        self.pdfs[stage_number] = _pdf

    # Add a method to generate stage paths on demand
    def add_stage(self, stage_number):
        """
        Dynamically generates paths for the specified stage number using PathTranslator.
        
        Args:
            stage_number (int): The stage number (0 for original, 1-3 for stages)
            
        Returns:
            dict: Dictionary containing the PathTranslator and paths for the specified stage
        """
        if stage_number == 0:
            stage_name = "original"
        else:
            stage_name = f"stage{stage_number}"
        
        # Create the stage directory path
        stage_dir = self.output_path.abs / Path(stage_name) / Path(self.sub_dir)
        
        # Ensure the directory exists
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a PathTranslator for the stage directory
        stage_path_translator = PathTranslator(
            absolute_path=stage_dir,
            base_path=self.output_path.abs.parent  # Use the same base path as output_path
        )
        
        # Create full file path
        file_abs_path = stage_dir / Path(self.file_name)
        
        self.stage_paths[stage_number] = {
            "translator": stage_path_translator,
            "abs_path": str(file_abs_path),
            "rel_path": str(stage_path_translator.rel / Path(self.file_name))
        }
        return str(file_abs_path)

    @classmethod
    def traverse_directory(cls, input_dir, output_dir) -> List["DirectoryFileNode"]:
        file_nodes: List[DirectoryFileNode] = []

        for root, dirs, files in os.walk(input_dir.abs):
            # Determine the relative path to maintain sub_dir info
            rel_path = os.path.relpath(root, input_dir.abs)
            if rel_path == '.':
                rel_path = ''

            for f in files:
                name, extension = os.path.splitext(f)
                file_type = extension.lstrip('.')  # e.g., 'txt' from '.txt'

                full_path = os.path.join(input_dir.abs, rel_path, f)

                # Create the node without stage_paths
                node = cls(
                    input_dir=input_dir.abs,
                    output_dir=output_dir.abs,
                    input_path=input_dir, 
                    output_path=output_dir,
                    sub_dir=rel_path,
                    file_name=f,
                    full_path=full_path,
                    file_type=file_type,
                    name=name,
                    uuid=str(uuid.uuid4())
                )
                
                # Handle the original file saving
                src_doc = fitz.open(full_path)
                original_abs_path = node.add_stage(0)

                src_doc.save(original_abs_path)
                orig_pdf = Pdf(original_abs_path, stage=0)
                node.store_metadata(0, orig_pdf.to_dict())
                node.pdfs[0] = orig_pdf
                file_nodes.append(node)

        return file_nodes

    def to_dict(self):
        return {k:v for k,v in asdict(self).items() if k not in ['time_util', 'pdfs']}

    def to_json(self):
        return safe_json(self.to_dict())

    @staticmethod
    def filter(
        table_ai: 'TableAi', 
        nodes: List["DirectoryFileNode"],
        extensions: Union[str, List[str]]
    ) -> List["DirectoryFileNode"]:
        """
        Returns a new list of file nodes from `nodes` whose file_type matches
        the specified extension(s).

        Args:
            nodes (List[DirectoryFileNode]): A list of file nodes to filter.
            extensions (Union[str, List[str]]): A file extension (e.g., 'txt')
                                                or a list of extensions (e.g., ['txt', 'py']).

        Returns:
            List[DirectoryFileNode]: A new list of DirectoryFileNode objects
                                     matching the given extension(s).
        """
        if isinstance(extensions, str):
            extensions = [extensions]  # Convert a single string to a list

        # Perform filtering
        return [node for node in nodes if node.file_type in extensions]

    def __repr__(self):
        """
        Custom representation. The dataclass would provide one automatically,
        but you can override to format it exactly how you want.
        """
        return (
            f"DirectoryFileNode("
            f"uuid={self.uuid!r}, "
            f"input_dir={self.input_dir!r}, "
            f"output_dir={self.output_dir!r}, "
            f"sub_dir={self.sub_dir!r}, "
            f"file_name={self.file_name!r}, "
            f"name={self.name!r}, "
            f"full_path={self.full_path!r}, "
            f"file_type={self.file_type!r}, "
            f"created_at={self.created_at!r}, "
            f"updated_at={self.updated_at!r}, "
            f"last_sync={self.last_sync!r})"
        )