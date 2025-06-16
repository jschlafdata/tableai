from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Any, Optional, Union, get_type_hints
from datetime import datetime
from abc import ABC, abstractmethod
import inspect

class FlowResultBase(BaseModel, ABC):
    """
    Base class for all flow result models. Provides generic functionality
    for summary reporting, display management, and metadata handling.
    """
    
    # === Core Flow Components (all flows have these) ===
    pdf_model: Optional['PDFModel'] = Field(
        default=None, 
        description="Reference to the PDFModel for display and analysis functionality"
    )
    
    processing_timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="When the processing was completed"
    )
    
    def get_summary_report(self) -> str:
        """Generate a formatted summary report using field annotations."""
        lines = []
        
        # Get all field info with annotations
        field_info = self.model_fields
        field_values = self.model_dump()
        
        # Extract sections based on field metadata
        sections = self._extract_sections_from_fields()
        
        for section_name, section_fields in sections.items():
            if section_name != "core":  # Skip core fields for now
                lines.append(f"# {section_name.upper().replace('_', ' ')}")
                
                for field_name in section_fields:
                    if field_name in field_values:
                        value = field_values[field_name]
                        field_desc = field_info[field_name].description or field_name.replace('_', ' ').title()
                        
                        if isinstance(value, str) and len(value) > 100:
                            lines.append(f"- {field_desc}: {len(value)} characters")
                        elif isinstance(value, (list, dict)):
                            lines.append(f"- {field_desc}: {len(value)} items")
                        elif isinstance(value, datetime):
                            lines.append(f"- {field_desc}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            lines.append(f"- {field_desc}: {value}")
                
                lines.append("\n" + "="*80 + "\n")
        
        return "\n".join(lines)
    
    def _extract_sections_from_fields(self) -> Dict[str, List[str]]:
        """Extract logical sections from field names and annotations."""
        sections = {
            "overview": [],
            "statistics": [],
            "configuration": [],
            "results": [],
            "images": [],
            "metadata": [],
            "core": []
        }
        
        for field_name, field_info in self.model_fields.items():
            # Skip pdf_model and processing_timestamp as they're core
            if field_name in ["pdf_model", "processing_timestamp"]:
                sections["core"].append(field_name)
                continue
                
            # Categorize based on field name patterns
            if any(keyword in field_name.lower() for keyword in ["overview", "goal", "description"]):
                sections["overview"].append(field_name)
            elif any(keyword in field_name.lower() for keyword in ["count", "_analyzed", "statistics"]):
                sections["statistics"].append(field_name)
            elif any(keyword in field_name.lower() for keyword in ["param", "config", "setting"]):
                sections["configuration"].append(field_name)
            elif any(keyword in field_name.lower() for keyword in ["image", "base64"]):
                sections["images"].append(field_name)
            elif any(keyword in field_name.lower() for keyword in ["result", "output", "regions"]):
                sections["results"].append(field_name)
            else:
                sections["metadata"].append(field_name)
        
        # Remove empty sections
        return {k: v for k, v in sections.items() if v}
    
    def display_images(self, **kwargs):
        """Display images using PDFModel if available, otherwise fallback to base64."""
        if self.pdf_model:
            # Use PDFModel's rich display functionality
            self.pdf_model.show(**kwargs)
        else:
            # Fallback to basic base64 display
            self._display_base64_images()
    
    def _display_base64_images(self):
        """Fallback method to display base64 images directly."""
        try:
            from IPython.display import display, Image as IPImage, HTML
            import base64
            
            # Find all base64 image fields
            field_values = self.model_dump()
            for field_name, value in field_values.items():
                if isinstance(value, str) and field_name.lower().endswith('image') and len(value) > 1000:
                    try:
                        img_bytes = base64.b64decode(value)
                        display(HTML(f"<h3>{field_name.replace('_', ' ').title()}</h3>"))
                        display(IPImage(data=img_bytes, width=800))
                    except Exception:
                        continue
                        
        except ImportError:
            print("📷 Images available (IPython required for display)")
            field_values = self.model_dump()
            for field_name, value in field_values.items():
                if isinstance(value, str) and field_name.lower().endswith('image') and len(value) > 1000:
                    print(f"   {field_name}: {len(value)} base64 characters")

class SectionMeta:
    """Helper class to define field sections for better organization."""
    def __init__(self, section: str, description: str = ""):
        self.section = section
        self.description = description

def section_field(section: str, description: str = "", **field_kwargs):
    """Create a field with section metadata for automatic report generation."""
    return Field(description=description, json_schema_extra={"section": section}, **field_kwargs)
