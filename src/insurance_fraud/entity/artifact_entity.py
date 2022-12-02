from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
          train_file_path:Path
          test_file_path:Path

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:Path
    invalid_train_file_path:Path
    valid_test_file_path:Path
    invalid_test_file_path:Path
    drift_report_file_path:Path