import pymongo 
import pandas as pd
import sys , os
from config import mongo_client

#creating database and collection name
database_name = "Ott"
collection_name = "Netflix"
#giving dataset file path
file_path = os.path.join(os.getcwd() , "netflix_dataset.csv")

if __name__=="__main__":
   #reading dataset as dataframe
   df= pd.read_csv(file_path)
   print(f"rows {df.shape[0]}  columns {df.shape[1]} ")
   client = pymongo.MongoClient(mongo_client)
   
   #dataset has to be eitherr json or dict format for dumping it into MongoDB
   
   dict_df = df.to_dict(orient="records")
   client[database_name][collection_name].insert_many(dict_df)
   print(">"*10," Dataset sucessfully stored inside a MongoDB ","<"*10)
   
   

