SELECT date_id, data_value
FROM observations
WHERE element = 'SNOW'
    AND data_value IS NOT NULL;
