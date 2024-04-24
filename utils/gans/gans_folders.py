from dataclasses import dataclass
from pathlib import Path

from utils_data.utils_folders import clean_folder, remove_folder


@dataclass
class FolderStructure:
    """
    Folder structure for GANs
    
    how to use:
    folder_structure=FolderStructure(trainning_output1=Path('SNGAN_training_output'))
    
    """
    
    def __init__(self,trainning_output) -> None:
        self._trainning_output:Path = trainning_output
        
        self.tensorboard_folder: Path = self._trainning_output / Path('Tensorboard_runs')
    
        self.trainning_output_checkpoints:Path   =  self._trainning_output / Path("checkpoints")
        self.trainning_final_model_folder:Path   =  self._trainning_output / Path("final_models") 

        self.trainning_model_interpolated_images =  self._trainning_output / Path("interpolated_images")

        self.trainning_images_generated:Path     =  self._trainning_output / Path("generator_images")
        self.one_stack_images_generated:Path     =  self.trainning_images_generated / Path("one_stack_of_images")
        self.stacks_images_generated:Path        =  self.trainning_images_generated / Path("stacks_of_images")
        self.debug_images_generated:Path        =  self.trainning_images_generated / Path("debug_images")
        self.test_images_generated:Path        =  self.trainning_images_generated / Path("test_images")

        
        
    def get_tensorboard_folder(self):
        return self.tensorboard_folder

    def remove_trainning_folder_output(self):
        remove_folder(self._trainning_output)    
    
    def get_list_of_folders(self):
        folders=[self.tensorboard_folder,
                 self.trainning_output_checkpoints, 
                 self.one_stack_images_generated,
                 self.debug_images_generated,
                 self.test_images_generated,
                 self.stacks_images_generated,
                 self.trainning_final_model_folder, 
                 self.trainning_model_interpolated_images]
    
        return folders
    
    def clean_folders(self):
        for folder in  self.get_list_of_folders():
            clean_folder(folder)
 
    def get_path_checkpoints_for_file(self, file_name):
        return self.trainning_output_checkpoints / file_name
    
    def create_folders(self):
        for folder in  self.get_list_of_folders():
            if not folder.is_dir():
                folder.mkdir(parents=True, exist_ok=True)
                
                
