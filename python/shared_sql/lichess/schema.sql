CREATE TABLE puzzles AS
SELECT
    PuzzleId AS puzzle_id,
    FEN AS fen,
    Moves AS moves,
    CAST(Rating AS INTEGER) AS rating,
    CAST(RatingDeviation AS INTEGER) AS rating_deviation,
    CAST(Popularity AS INTEGER) AS popularity,
    CAST(NbPlays AS INTEGER) AS num_plays,
    Themes AS themes,
    GameUrl AS game_url,
    OpeningTags AS opening_tags,
    make_timestamp_ms(TRY_CAST(DailyDate AS BIGINT)) AS daily_date
FROM read_csv_auto('{directory}/puzzles.csv', header=true);
