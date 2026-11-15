SELECT date_id, data_value
FROM observations
WHERE element = 'TMAX'
    AND data_value IS NOT NULL;
