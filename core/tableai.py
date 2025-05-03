from core.paths import Paths
from core.log import critical, error, warning, info, debug
from core.config import CONFIG
from core.constants import (
    OUTPUT_DIRECTORY
)
import argparse
from pathlib import Path
from pdf.file import DirectoryFileNode
from extract.stages import Stage1, Stage2
from data_writers.export import DictWriter
from core.sync import sync_directories

class TableAi(Paths):
    def __init__(
            self,
            input_dir: str,
            output_dir: str = OUTPUT_DIRECTORY
    ):
        super().__init__(input_dir, output_dir)
        self._init_log()
        self.info("Table AI Extraction Beginning!")
        self.config = CONFIG.root(self.root_dir)
        self.info(
            f"Project is running with the following settings:\n"
            f"{CONFIG.pretty(self.config)}"
        )
        self._build_dirs()
    
    def _init_log(self):
        self.critical = critical
        self.error = error
        self.warning = warning
        self.info = info
        self.debug = debug
    
    def _build_dirs(self):
        Path(self.state_path.abs).mkdir(parents=True, exist_ok=True)
    
    def process(self):

        file_nodes = DirectoryFileNode.traverse_directory(input_dir=self.input_path,output_dir=self.output_path)
        pdf_nodes = DirectoryFileNode.filter(self, file_nodes, "pdf")
        file_node = self.process_stage1(pdf_nodes[0])
        file_node = self.process_stage2(file_node)
        self.sync([file_node])

    def process_stage1(self, file_node):
        stage1_configs = self.config.get('extract', {}).get('stage1', {})
        file_node.add_stage(1)
        extractor = Stage1(
            directory_file_node=file_node,
            header_bound=stage1_configs.get('header_bound', 100),
            footer_bound=stage1_configs.get('footer_bound', 100),
            min_occurrences=stage1_configs.get('min_occurrences', 2)
        )
        stage1_metadata = extractor.run()
        file_node.store_metadata(1, stage1_metadata)
        return file_node

    def process_stage2(self, file_node):
        file_node.add_stage(2)
        extractor = Stage2(file_node, parameters=self.config, merge=True)
        grouped_tables, inverse_tbl_bounds, occupied_bounds, table_key_dict, page_width, page_height, merged_tables = extractor.run()
        stage2_meta = {
            'page_width': page_width, 
            'page_height': page_height, 
            'tables': grouped_tables, 
            'inverse_tbl_bounds': inverse_tbl_bounds, 
            'occupied_bounds': occupied_bounds, 
            'table_key_dict': table_key_dict,
            'merged_tables': merged_tables
        }
        file_node.store_metadata(2, stage2_meta)
        return file_node
    
    def sync(self, file_nodes):

        state=[]
        for node in file_nodes:
            node_meta = node.to_dict()
            state.append(node_meta)

        DictWriter(state, self.state_directory, 'metadata.json').write()
        sync_directories(self.output_dir, self.react_dir)
        return


def main():
    parser = argparse.ArgumentParser(description='Build and upload wheel if changes are detected.')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Directory for docs to process.')
    args = parser.parse_args()

    tableai = TableAi(input_dir=args.input_dir)
    # print(dir(tableai))

    tableai.process()


if __name__ == "__main__":
    main()
