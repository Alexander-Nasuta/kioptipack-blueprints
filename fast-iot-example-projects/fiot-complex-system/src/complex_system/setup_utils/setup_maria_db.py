from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace 'your_username', 'your_password', 'your_host', 'your_port', and 'your_database' with your MariaDB credentials
DATABASE_URL = "mysql+pymysql://fiot:fiotdev123@localhost:3306/TestDB"
asd = "fiot:fiotdev123@localhost:3306/TestDB"

engine = create_engine('mysql+pymysql://'+ asd, echo=True)

# Create a declarative base class
Base = declarative_base()

# Define a simple User model
class User(Base):
    __tablename__ = 'users3'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(100), unique=True)

# Create the tables in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# Example: Inserting a new user into the 'users' table
new_user = User(username='john_doe', email='john@example.com')
session.add(new_user)
session.commit()

# Querying the 'users' table
users = session.query(User).all()
for user in users:
    print(f"User ID: {user.id}, Username: {user.username}, Email: {user.email}")

# Close the session
session.close()

if __name__ == '__main__':
    print("run `CREATE DATABASE TestDB` before running this application")
    pass