CREATE TABLE stations AS
SELECT
    TRIM(SUBSTR(line, 1, 11)) AS station_id,
    CAST(NULLIF(REGEXP_REPLACE(TRIM(SUBSTR(line, 13, 9)), '[^0-9.-]', '', 'g'), '') AS DOUBLE) AS latitude,
    CAST(NULLIF(REGEXP_REPLACE(TRIM(SUBSTR(line, 22, 10)), '[^0-9.-]', '', 'g'), '') AS DOUBLE) AS longitude,
    CAST(NULLIF(REGEXP_REPLACE(TRIM(SUBSTR(line, 32, 6)), '[^0-9.-]', '', 'g'), '') AS DOUBLE) AS elevation,
    TRIM(SUBSTR(line, 39, 2)) AS state,
    TRIM(SUBSTR(line, 41, 31)) AS name
FROM read_csv_auto('{directory}/stations.txt', header=false, columns={'line': 'VARCHAR'}, all_varchar=true);

CREATE TABLE observations AS
SELECT
    ID AS station_id,
    TRY_CAST(DATE AS INTEGER) AS date_id,
    ELEMENT AS element,
    TRY_CAST(DATA_VALUE AS INTEGER) AS data_value,
    M_FLAG AS m_flag,
    Q_FLAG AS q_flag,
    S_FLAG AS s_flag
FROM read_csv_auto('{directory}/*.csv', header=true, all_varchar=true, ignore_errors=true);
