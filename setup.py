from setuptools import find_packages,setup
from typing import List

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements()->List[str]:
    
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
    #removing hyphen e from requirement.txt 
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list

#creating our own package src , all requirements and dependency install inside a src.egg , we have to mention __init__ inside our folder

setup(
    name="src",
    version="0.1",
    author="RJ",
    author_email="rajat.k.singh64@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)