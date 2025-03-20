#!/bin/bash
#Edit with your user and db name
user=username
database=dbname
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
	CREATE USER $user;
	CREATE DATABASE $database;
	GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
EOSQL

