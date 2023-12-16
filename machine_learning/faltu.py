import sqlite3
conn=sqlite3.connect("D:\\demo\\alpha.db")
#creating the cursor here
cur=conn.cursor()
names_list=[
    ("rohit","gupta"),
    ("hello","world"),
    ("jio","5g")
]
cur.executemany("INSERT INTO altt (first_name,second_name) VALUES (?,?)",names_list)
#close the db objects
cur.close()
conn.close()    
#there are four types of data in sqlite 
#integer , real , text , blob and we also have a type called NULL
#          float  string byte                                None
