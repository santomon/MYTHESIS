from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Env(BaseSettings):
    proton_density_dicom_path: str
    proton_density_nifti_path: str


env = Env()
