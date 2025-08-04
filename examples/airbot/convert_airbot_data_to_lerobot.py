import h5py
import cv2
import shutil
from pathlib import Path
import shutil
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "transform2lerobot_data/multi_task"
RAW_DATASET_NAMES = [
    "h5_files_base",
    # "h5_files_0311",
    # "h5_files_0314",
]


def main(*, push_to_hub: bool = False):
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="airbot",
        fps=10,
        features={
            "top_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    BASE_DIR = "/data2/airbot/single/processed/agent"
    for raw_dataset_name in RAW_DATASET_NAMES:
        
        folder_path = Path(BASE_DIR) / raw_dataset_name
        h5_files = list(folder_path.glob("*.h5"))
        for h5_path in h5_files:
            
                with h5py.File(h5_path, 'r') as hdf_file:
                    group = hdf_file

                    for idx in range(len(group['action'])):
                        
                        dataset.add_frame({
                            "top_image": cv2.imdecode(group['top_image'][idx], cv2.IMREAD_COLOR),
                            "wrist_image": cv2.imdecode(group['wrist_image'][idx], cv2.IMREAD_COLOR),
                            "state": group['proprio'][()][idx].astype('float32'),
                            "actions": group['action'][()][idx].astype('float32'),
                            # "task": group['language'][()].decode('utf-8'),
                        })
                    
                    dataset.save_episode(task=group['language'][()].decode('utf-8'))

    dataset.consolidate(run_compute_stats=True)

if __name__ == "__main__":
    tyro.cli(main)
    
    
