SELECT rating // 10, theme
FROM puzzles,
    unnest(string_to_array(themes, ' ')) AS t(theme)
WHERE opening_tags LIKE '%Caro-Kann%'
    AND themes != '';
