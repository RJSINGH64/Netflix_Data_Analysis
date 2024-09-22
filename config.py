import os  , sys
from dataclasses import dataclass
from dotenv import load_dotenv
import pymongo
print("Loading .env")
load_dotenv()



@dataclass
class EnvironmentVariable:
    pymongo_url:str = os.getenv("Mongo_Url")


env = EnvironmentVariable()    
print(f">>>> Getting Mongo Url <<<<<")
mongo_client=env.pymongo_url

