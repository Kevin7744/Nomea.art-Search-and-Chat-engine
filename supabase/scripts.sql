-- Supabase AI is experimental and may produce incorrect answers
-- Always verify the output before executing

drop function if exists match_documents (vector (1024), jsonb);

create function match_documents (
  query_embedding vector (1024),
  filter jsonb default '{}'
) returns table (id integer) language plpgsql as $$
begin
  return query
  select
    posts.id::integer as id
  from posts
  order by posts.embeddings <=> query_embedding;
end;
$$;

-- ALTER TABLE public.posts
-- ADD COLUMN embeddings vector(1024);

-- ALTER TABLE public.posts
-- ADD COLUMN images bytea;

-- ALTER TABLE public.posts
-- ADD COLUMN image_embeddings bytea NULL;

-- CREATE OR REPLACE FUNCTION euclidean_distance(arr1 FLOAT8[], arr2 FLOAT8[])
-- RETURNS FLOAT AS $$
-- DECLARE
--     sum FLOAT8 := 0;
--     i INT;
-- BEGIN
--     FOR i IN 1..array_length(arr1, 1) LOOP
--         sum := sum + POWER(arr1[i] - arr2[i], 2);
--     END LOOP;
--     RETURN SQRT(sum);
-- END;
-- $$ LANGUAGE plpgsql IMMUTABLE;

CREATE TYPE match_result AS (
  id BIGINT,
  distance FLOAT
);


CREATE OR REPLACE FUNCTION match_image(query_embedding FLOAT8[])
RETURNS TABLE (id BIGINT, distance FLOAT) AS $$
DECLARE
    -- Variable to hold each post record during the loop
    rec RECORD;
    -- Array to hold the results as match_result type
    results match_result[] := ARRAY[]::match_result[];
    -- Variables for calculation
    element FLOAT8;
    diff FLOAT8;
    sum FLOAT8;
    dist FLOAT8;
    i INT;
BEGIN
    -- Loop through each post with embeddings
    FOR rec IN SELECT id, new_image_embeddings FROM public.posts WHERE new_image_embeddings IS NOT NULL LOOP
        sum := 0;
        -- Calculate Euclidean distance
        FOR i IN 1..array_length(query_embedding, 1) LOOP
            element := rec.new_image_embeddings[i];
            diff := query_embedding[i] - element;
            sum := sum + (diff * diff);
        END LOOP;
        dist := SQRT(sum);
        
        -- Store the id and distance in the results array as a match_result type
        results := array_append(results, ROW(rec.id, dist)::match_result);
    END LOOP;
    
    -- Unnest the results array to return it as a set of rows, ordered by distance
    RETURN QUERY SELECT r.id, r.distance FROM unnest(results) AS r(id, distance) ORDER BY r.distance;
END;
$$ LANGUAGE plpgsql;




-- Step 1: Delete the current image_embeddings column from the posts table
-- ALTER TABLE public.posts DROP COLUMN image_embeddings;

-- Step 2: Add a new column for storing image embeddings
-- ALTER TABLE public.posts ADD COLUMN new_image_embeddings BYTEA;

-- ALTER TABLE public.posts
-- ALTER COLUMN new_image_embeddings TYPE FLOAT8[] USING new_image_embeddings::FLOAT8[];


-- ALTER TABLE public.posts
-- ALTER COLUMN new_image_embeddings TYPE FLOAT8[]
-- USING new_image_embeddings::FLOAT8[];

-- ALTER TABLE public.posts DROP COLUMN new_image_embeddings;
-- ALTER TABLE public.posts ADD COLUMN new_image_embeddings FLOAT8[];

-- ALTER TABLE public.posts
-- ALTER COLUMN new_image_embeddings TYPE FLOAT8[]
-- USING NULL; -- This drops the current data; be cautious!

