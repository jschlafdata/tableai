from pathlib import Path

class PathManager:
    project_root = Path(__file__).resolve().parent.parent.parent  # Adjust if needed
    synced_dir = project_root / ".synced" / "dropbox" / "pdf"
    output_dir = project_root / ".processing" / "outputs"
    recovery_dir = project_root / ".recovery"
    llm_dir = project_root / ".llm" / "outputs"

    @staticmethod
    def get_stage_dir(stage: int, file_type: str = None) -> Path:
        if stage == 0:
            return PathManager.synced_dir
        return PathManager.output_dir / f"stage{stage}"

    @staticmethod
    def get_stage_pdf_path(uuid: str, file_type: str, stage: int) -> Path:
        return PathManager.get_stage_dir(stage) / f"{uuid}.{file_type}"

    @staticmethod
    def get_recovery_path(uuid: str, file_type: str, stage: int) -> Path:
        return PathManager.recovery_dir / file_type / f"stage{stage}" / f"{uuid}.{file_type}"

    @staticmethod
    def get_rel_path(abs_path: Path) -> str:
        return str(abs_path.relative_to(PathManager.project_root))

    @staticmethod
    def get_fastapi_mount_path(stage: int, recovered: bool=False) -> str:
        if stage == 0:
            if recovered == True:
                return "/files/recovered/stage0"
            else:
                return "/files/stage0"
        return f"/files/extractions/stage{stage}"

    @staticmethod
    def all_mount_configs():
        return {
            "/files/stage0": PathManager.synced_dir,
            "/files/extractions": PathManager.output_dir,
            "/files/recovered": PathManager.recovery_dir,
            "/files/llm": PathManager.llm_dir
        }