SELECT num_plays // 1000, rating // 10
FROM puzzles
WHERE num_plays > 1000
ORDER BY num_plays DESC;
