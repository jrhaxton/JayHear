# Flask Project

## Python Virtual Environment
Create environment
```
python3 -m venv ./path/to/env
```

Activate environment
```
source ./path/to/env/bin/activate
```

Install requisite packages
```
pip install -r ./requirements.txt
```

## Set Up Flask Environment
Run development server
```
flask run
```

If this doesn't work, make sure to run
```
export FLASK_APP=app
export FLASK_ENV=development
```

## Building the Test Database
```
python3
>>> from app import db
>>> db.create_all()
```
**It is required to delete the test.db file and rebuild it using these commands when modifying the models**

## Interacting with Database
**Line 14 in app.py must be updated to the file location of the test.db file**

Open test.db in SQLite3
```
sqlite3 test.db
```

Show tables in database
```
.tables
```

Show entries in a table
```
select * from table;
```

Delete all entries in table
```
delete from table;
```





