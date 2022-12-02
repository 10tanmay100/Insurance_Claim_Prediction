#use to create proper package in python
from setuptools import find_packages,setup
from typing import List

#defining the function which will be responsible for installing the all requirements inside txt file
def get_install_requirements_file()->List[str]:
          '''
                    installing all the mentioned requirements
          '''
          with open("requirements.txt","r") as requirements_file:
                    return requirements_file.readlines()




setup(
          name="Insurance_Fraud",
          version="0.0.2",
          author="Tanmay Chakraborty",
          author_email="chakrabortytanmay744@gmail.com",
          packages=find_packages("insurance_fraud"),
          install_requires=get_install_requirements_file()
)










