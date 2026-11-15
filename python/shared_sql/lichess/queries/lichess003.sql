SELECT TRY_CAST(split_part(fen, ' ', 6) AS INTEGER) AS move_number, rating // 10
FROM puzzles
WHERE TRY_CAST(split_part(fen, ' ', 6) AS INTEGER) < 20;
