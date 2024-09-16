import pymongo 
import pandas as pd
import sys , os

#creating database and collection name
database_name = "ott"
collection_name = "Netflix"
#giving dataset file path
file_path = r"E:\PYTHON PROJECTS\V-S Code Projects\Netflix_Data_Analysis\netflix_dataset.csv"

if __name__=="__main__":git 
   #reading dataset as dataframe
   df= pd.read_csv(file_path)
   print(f"rows{df.shape[0]} columns{df.shape[1]}\n*4")
   #client = pymongo.MongoClient("mongodb+srv://yourusername:yourpass@cluster0.owfzau8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
   #dataset has to be eitherr json or dict format for dumping it into MongoDB
   
   dict_df = df.to_dict(orient="records")
   #client[database_name][collection_name].insert_many(dict_df)
   
   

