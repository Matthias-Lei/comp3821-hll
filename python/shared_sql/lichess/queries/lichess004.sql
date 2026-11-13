SELECT rating // 10, themes
FROM puzzles
WHERE array_length(string_to_array(moves, ' ')) >= 4
  AND moves IS NOT NULL AND moves != '';
