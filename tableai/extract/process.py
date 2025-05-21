
from backend.models.backend import (
    FileExtractionResult, 
    ClassificationLabel, 
    FileNodeRecord, 
    PDFClassifications, 
    DropboxSyncRecord
)

from tableai.nodes.file_node import DirectoryFileNode
import json
import hashlib
from copy import deepcopy
from typing import List, Dict, Any
from tableai.core.paths import PathManager
from tableai.data_loaders.files import FileReader

### FROM ORIGINAL DEPLOYMENT ###
from tableai.extract.stages import Stage1, Stage2, Stage3
from tableai.extract.helpers import Stitch

import os
import fitz
import shutil
import tempfile
from typing import Dict, Any, List

class DropboxRegisterService:

    def __init__(self, instance: 'DBManager'):
        self.instance = instance 
        self.existing_nodes_by_uuid = {}
        self._register_nodes()

    def _register_nodes(self):
        nodes = self.instance.run_op(
            FileNodeRecord,
            operation="get",
            filter_by={"source": "dropbox"},
            columns=[
                "uuid", "file_name", "local_path", 
                "stage_paths_json", "completed_stages_json", 
                "extraction_metadata_json" 
            ]
        )
        self.existing_nodes_by_uuid = {}
        for node in nodes:
            _stage0_meta={}
            stage_paths = json.loads(node.stage_paths_json or "{}")
            extraction_metadata = json.loads(node.extraction_metadata_json or "{}")
            completed_stages = json.loads(node.completed_stages_json or "{}")
            if completed_stages:
                completed_stages = [int(x) for x in completed_stages]
            stored_stages = {int(x) for x in stage_paths.keys()}
            stage_meta = [
                    extraction_metadata.get(f"stage{x}", {}) for x in stage_paths.keys()
            ]
            if stage_meta:
                stage0 = stage_meta[0]
                _stage0_meta = {tag: stage0.get(tag) for tag in ['meta_tag', 'recovered', 'corrupt_flag']}
            self.existing_nodes_by_uuid[node.uuid] = {
                    'file_name': node.file_name,
                    'local_path': node.local_path, 
                    'stored_stages': stored_stages,
                    'stage0_metadata': _stage0_meta,
                    "completed_stages": completed_stages
            }
    
    def add(self, file_id=None, directory=None, node_list=None, force=False, stage=0):
        """
        Add new sync nodes from:
          - A list of node objects (node_list), OR
          - All nodes in a particular directory, OR
          - A specific file_id (UUID).

        :param file_id: Specific file UUID to register
        :param directory: Name/path to filter DropboxSyncRecords by
        :param node_list: List of pre-fetched node objects to register
        :param force: If True, re-register nodes even if stage is already completed
        :param stage: Processing stage (int)
        :return: A single DirectoryFileNode, a list of them, or None
        """
        # Ensure node_list is always a list
        synced_nodes = []

        # 1) Register from explicit node_list
        if node_list:
            for node in node_list:
                synced_node = self._register_sync_node(node, stage=stage)
                if synced_node:
                    synced_nodes.append(synced_node)

        # 2) Register all records from a directory
        if directory:
            print(f"Processing sync for directory: {directory} | Forcing: {force}")
            synced_records = self.instance.run_op(DropboxSyncRecord, operation="get")
            # Filter only those records matching directory
            matching = [
                r for r in synced_records
                if directory.lower() in r.path_lower.lower()
            ]

            if not force:
                # Exclude those already completed at this stage
                matching = [
                    r for r in matching
                    if r.dropbox_safe_id not in self.existing_nodes_by_uuid
                    or stage not in self.existing_nodes_by_uuid[r.dropbox_safe_id]['completed_stages']
                ]

            for record in matching:
                synced_node = self._register_sync_node(record, stage=stage)
                if synced_node:
                    synced_nodes.append(synced_node)

        # 3) Register a single record by file_id
        single_synced_node = None
        if file_id:
            print(f"Looking up file_id: {file_id}")
            sync_records = self.instance.run_op(
                DropboxSyncRecord, operation="get", filter_by={"dropbox_safe_id": file_id}
            )
            if sync_records:
                single_synced_node = self._register_sync_node(sync_records[0], stage=stage)

        # Perform any post-classification logic
        self._sync_post_classification()

        # Return either a single node (if requested by file_id) or the list
        return single_synced_node if single_synced_node else synced_nodes

    def get_node(self, file_id=None, file_node=None):
        if file_id:
            file_node = self._file_node_get(file_id)
            directory_file_node = DirectoryFileNode.from_record(file_node)
            return directory_file_node

    def _get_node_mappings(self, node_label):
        try:
            classified_meta, base_pipeline_conf = self._load_classified_meta()
            process_meta = classified_meta.get(node_label)
            tables_search_config = expand_table_combinations(process_meta, node_label)
            return base_pipeline_conf, tables_search_config
        except:
            return None, None
    
    def _load_classified_meta(self):
        llm_config_file = PathManager.llm_dir / "pdf_schema.yml"
        base_conf = PathManager.project_root / "settings.yml"
        base_llm_config = FileReader.yaml(llm_config_file)
        base_pipeline_conf = FileReader.yaml(base_conf)
        return base_llm_config, base_pipeline_conf
    
    def _get_node_label(self, directory_file_node):
        try:
            classification_maps = self._get_classification_maps()
            meta_tag = directory_file_node.extraction_metadata.get('stage0', {}).get('meta_tag')
            auto_label = directory_file_node.auto_label
            node_label = classification_maps.get(meta_tag, None)
            if auto_label:
                node_label = auto_label
            return node_label
        except:
            return None

    def _get_classification_maps(self):
        classification_labels = self.instance.run_op(ClassificationLabel, operation="get")
        classification_maps = {label.classification: label.label for label in classification_labels}
        return classification_maps
    
    def _register_sync_node(self, record, stage=0):
        """
        Helper to create/register a DirectoryFileNode from a DropboxSyncRecord
        and store it in the database.
        """
        if not record:
            return None

        file_node = DirectoryFileNode.from_sync_record(
            record=record,
            source='dropbox',
            current_stage=stage
        )
        file_node.register(db=self.instance)
        return file_node

    def sync_node_add(self, node=None, node_list=None, stage=0):
        """
        Alternative entry point if you only have node(s) (without directory/file_id).
        """
        synced_nodes = []
        # Convert to list if a single item is passed
        if isinstance(node, list):
            node_list = node
            node = None

        if node_list:
            for n in node_list:
                synced = self._register_sync_node(n, stage=stage)
                if synced:
                    synced_nodes.append(synced)
        elif node:
            synced = self._register_sync_node(node, stage=stage)
            if synced:
                synced_nodes.append(synced)
        return synced_nodes

    def _sync_node_get(self, file_id, stage=0):
        """
        Fetch a DropboxSyncRecord by file_id, build and register a DirectoryFileNode.
        """
        sync_record = self.instance.run_op(
            DropboxSyncRecord, operation="get", filter_by={"dropbox_safe_id": file_id}
        )
        if not sync_record:
            return None
        return self._register_sync_node(sync_record[0], stage=stage)

    def _file_node_get(self, file_id):
        """
        Retrieve an existing FileNodeRecord by UUID from the DB (not necessarily re-register).
        """
        records = self.instance.run_op(
            FileNodeRecord, operation="get", filter_by={"uuid": file_id}
        )
        return records[0] if records else None

    def _sync_post_classification(self):
        """
        Called after we finish registering nodes. 
        Here you can handle classification merges or updates in bulk.
        """
        # Possibly refresh self.existing_nodes_by_uuid or something similar.
        self._register_nodes()  # not shown here, but presumably updates your local state

        meta_updates = []
        for node_uuid, meta in self.existing_nodes_by_uuid.items():
            meta_tag = meta.get('stage0_metadata', {}).get('meta_tag')
            if meta_tag:
                meta_updates.append(PDFClassifications(file_id=node_uuid, classification=meta_tag))

        if meta_updates:
            self.instance.run_op(PDFClassifications, operation="merge_many", data=meta_updates)



def inject_optional_cols(column_indexes: dict, optional_cols: dict) -> dict:
    """
    Inject optional columns into column_indexes at specified indices,
    shifting existing ones as needed.
    """
    for insert_idx in sorted(optional_cols.keys()):
        col_name = optional_cols[insert_idx]
        updated = {}
        for k in sorted(column_indexes.keys(), reverse=True):
            if k >= insert_idx:
                updated[k + 1] = column_indexes[k]
            else:
                updated[k] = column_indexes[k]
        column_indexes = updated
        column_indexes[insert_idx] = col_name
    return dict(sorted(column_indexes.items()))

def expand_table_combinations(config: Dict[str, List[Dict[str, Any]]], source_name=None) -> List[Dict[str, Any]]:
    """
    Generate all possible table variations with:
    - optional column insertions
    - table_name and totals string variants (pipe-delimited)

    Args:
        config: schema config (e.g., { "Fiserv": [table1, table2, ...] })

    Returns:
        List of tables with:
        - source
        - table_name (variant)
        - columns (list of strings)
        - totals (variant)
        - table_key (unique hash)
    """
    all_tables = []
    if config:
        for index, table in enumerate(config):
            base_columns = table.get("columns", {})
            optional_variants = table.get("optional_cols", [])
            totals_raw = table.get("totals", "")
            table_name_raw = table.get("table_name", "")

            # Handle table name variants
            table_name_variants = [name.strip() for name in table_name_raw.split("|")]
            total_variants = [t.strip() for t in totals_raw.split("|")] if isinstance(totals_raw, str) else [totals_raw]

            # Build column combinations (base + optional inserts)
            column_variants = [list(base_columns.values())]

            for opt in optional_variants:
                injected = inject_optional_cols(deepcopy(base_columns), opt)
                column_variants.append(list(injected.values()))

            # Cross product of all combinations
            for table_name in table_name_variants:
                for col_set in column_variants:
                    for total in total_variants:
                        tbl_meta = {
                            "source": source_name,
                            "table_name": table_name,
                            "columns": col_set,
                            "totals": total
                        }
                        table_key = hashlib.sha256(
                            '|'.join(col_set + [total, table_name]).encode("utf-8")
                        ).hexdigest()
                        tbl_meta["table_key"] = table_key
                        all_tables.append(tbl_meta)

        return all_tables
    else:
        return