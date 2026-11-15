SELECT station_id, data_value
FROM observations
WHERE element = 'TMIN'
    AND data_value < 0;
