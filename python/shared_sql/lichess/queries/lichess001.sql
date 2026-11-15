SELECT rating // 5, theme
FROM puzzles,
    unnest(string_to_array(themes, ' ')) AS t(theme)
WHERE themes != ''
    AND rating BETWEEN 1500 AND 2000;
