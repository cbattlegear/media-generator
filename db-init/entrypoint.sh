#!/bin/bash
# Wait for SQL Server to start and then run the init script

# Start SQL Server in background
/opt/mssql/bin/sqlservr &

# Wait for SQL Server to be ready
echo "Waiting for SQL Server to start..."
sleep 30

# Run the init script
for i in {1..50};
do
    /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -d master -i /docker-entrypoint-initdb.d/init-db.sql -C
    if [ $? -eq 0 ]
    then
        echo "Database initialization complete."
        break
    else
        echo "SQL Server not ready yet, waiting..."
        sleep 2
    fi
done

# Keep container running
wait
