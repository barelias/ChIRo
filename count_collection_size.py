from pymongo import MongoClient


client = MongoClient(host="localhost", port=27017, username="root", password="MongoDB2019!")
collection = client.ttc.ccm
print(collection.find_one({ "df_type": "validation" }))
print(collection.count_documents({ "df_type": "validation" }))
print(collection.find_one({ "df_type": "test" }))
print(collection.count_documents({ "df_type": "test" }))
print(collection.find_one({ "df_type": "train" }))
print(collection.count_documents({ "df_type": "train" }))
