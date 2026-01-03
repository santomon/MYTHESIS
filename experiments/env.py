from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Env(BaseSettings):
    proton_density_dicom_path: str
    proton_density_nifti_path: str
    proton_density_mask_nifti_path: str

    perfusion_test_dicom_path: str
    perfusion_test_nifti_path: str
    perfusion_test_mask_nifti_path: str


    linear_params_path: str


env = Env()
