CREATE TABLE stops AS
SELECT * FROM read_csv_auto('{directory}/csv/*.csv', header=true, all_varchar=true, ignore_errors=true);
