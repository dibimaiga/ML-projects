#The main aim is to read the dataset from some data source. That data source can be created from the big data team
#or from the cloud Team or may be reading from the live stream data
#So we will read , then split the dataset and then transform it


import os 
import sys
#The reason we're importing these is because we'll use our customException
from src.exception import CustomException
from src.logger import logging

import pandas as pd #we have a dataframe

from sklearn.model_selection import train_test_split

from dataclasses import dataclass #(used to create class variables)

# In order to implement the dataingestion component there should be some input required
#for exemple where will I save the train dataset or test dataset or even the raw data 

@dataclass # because i will just create some class variables 
class dataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

# here are those inputs above. Now we know where to save train data and all

class DataIngestion: #here w're not just gonna crete class variables, there will some functions(that's why we don't need the (@dataclass)and so we will create a constructor part )
    def __init__(self):
        self.ingestion_config = dataIngestionConfig() # to initialize the path(all 3 in this single variable)
#Now I'll write my own fucntion
    def initiate_data_ingestion(self):
        """
        Here we'll focus on reading the data from a database
        :param self: Description
        """
        logging.info("Entered the data ingestion component or method")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the data as dataframe")

            #Now let's create the folders(artifacts) (we know the paths already)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #I have to combine the directory pathname inside

            #let's save the raw data into 'raw_data_path' 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,train_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
#return these two(train_data_path and test_data_path) , so that our data transfotmationcomponent will take it and use it
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion() #initialization(data ingestion object)
    obj.initiate_data_ingestion()



