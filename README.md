# Battlecabbage Media Generator

This respository is host to a media generation tool that is part of a larger [Battlecabbage Media](https://github.com/Battlecabbage-Media) project. This tools makes use of multiple Azure OpenAI models.

- Azure Open AI

    - GPT-4

    - GPT-4 Turbo for Vision capabilities

    - Dalle3

The result is complete movie with title, tagline, description, content rating, critic review and more. In addition a poster is created to match the movie allowing for what is effectively an automatic movie maker. All of this data is largely utilized for [Battlecabbage Media](https://battlecabbage-movies.azurewebsites.net)

## Trailer hosting

The API accepts existing MP4 trailers; it does not generate or transcode video.
Upload with `PUT /movies/{movie_id}/trailer`, a multipart `file` field with content
type `video/mp4`, and the same `X-Api-Key` used for poster uploads:

```powershell
curl.exe -X PUT "http://localhost:8000/movies/42/trailer" `
  -H "X-Api-Key: YOUR_API_KEY" `
  -F "file=@C:\Videos\trailer.mp4;type=video/mp4"
```

The response is the updated movie, including a relative URL such as
`"trailer_url": "/trailers/movie_42_<unique-id>.mp4"`. Movie listings and detail
responses also include `trailer_url` (`null` until a trailer is uploaded).
Resolve relative URLs against the API origin, not a separately hosted frontend.
Like posters, trailers are **publicly readable without an API key**.
`GET` and `HEAD` are supported, including HTTP byte ranges for seeking.

Uploads are limited to **250 MiB per file** by default. Set the positive integer
`TRAILER_MAX_SIZE_MIB` in the API environment to change the cap, then restart the
API. A request-level limit allows an additional 1 MiB for multipart overhead and
is enforced while receiving the body, even without `Content-Length`. Invalid
media returns `422`, unsupported content types return `415`, oversized uploads
return `413`, and an unknown movie returns `404`.

Videos are checked with `ffprobe` for a readable MP4 container and a video stream.
For broad browser compatibility, export H.264 video with AAC audio and use MP4
"fast start" (`ffmpeg -movflags +faststart`). Validation does not transcode files
or guarantee that every frame is decodable. MP4s with other codecs may not play
in every browser.

### Deployment and existing databases

**Apply the schema update before deploying the updated API.** For Docker Compose,
the idempotent initialization script upgrades existing databases as well as
creating new ones:

```powershell
docker compose run --rm db-init
docker compose up -d --build api
```

For an independently managed SQL Server, run
`db-init\migrations\001-add-trailer-url.sql` against the movie database using your
normal database deployment tooling. It adds nullable `movies.trailer_url` and
can safely be rerun; existing movie rows are preserved.

Docker Compose persists video storage with `./trailers:/app/trailers`, and the
API image includes FFmpeg. Outside Docker, install FFmpeg with `ffprobe` on the
API process's `PATH`, and install the updated `requirements.txt` (including
Starlette's byte-range response support). Missing `ffprobe` returns `503` on
otherwise valid MP4 uploads.

Files are staged in `trailers\.uploads` and moved atomically into
`trailers\public` only after validation. Only `public` is mounted at `/trailers`.
A failed upload leaves the previous trailer URL unchanged. Each successful
upload gets a new URL; **previous versions are retained** so cached links,
in-progress playback, and concurrent uploads remain valid. Plan storage
retention/cleanup accordingly. All API instances must share this storage when
running on multiple hosts.

If using a reverse proxy, allow at least 251 MiB request bodies for the default
cap (adjust with the configured cap), set suitable upload/read timeouts, and
preserve `Range`, `If-Range`, `Content-Range`, and `Accept-Ranges` headers.
Do not buffer whole trailer responses in application memory.

### Trailer tests

Install the API requirements and the existing development dependencies
(`pip install -r requirements.txt` and `pip install -e ".[dev]"`), with FFmpeg
and `ffprobe` on `PATH`, then run:

```powershell
python -m pytest tests\test_trailers.py
```

The tests use a temporary SQLite database and a tiny generated MP4, not a live
SQL Server or an AI backend.

## Media Generation

![Media Generation Flow](assets/images/media_generation_flow.jpeg)

### The Flow:

1) Prompt Building

    - [Media Generator](library-management/generators/media_generator.py) heavily utilitizes the [prompts.json](library-management/templates/prompts.json) containing handwritten prompts for various Azure OpenAI model endpoints that are utilized throughout the generation. The prompts are written to be dynamic and are filled in at random from the category json files (e.g. [genres.json](library-management/templates/genres.json)). Its a movie maker madlib!

2) Generating Content

    - The intitial movie prompt is sent to GPT 4 to have it generate a movie complete with its title, tagline, and plot from dynamic built prompt as a movie completion.

    - The movie completion is sent to a GPT 4 endpoint, asking it to create critic review from the movie details.

    - The movie completion is also sent to a GPT 4 endpoint, asking it to create an image prompt for Dalle-3 to create a movie poster based upon the movie completion details.

    - The image completion is sent to Dalle-3 to create a poster for the movie.

    - The generated poster image is sent to GPT-4 Vision, asking it to assess the poster, if it needs to have a title added, if so where and what font color.

3) Saving Movie

    - Based upon the response from GPT-4 4, the title is possibly added to the image and saved alongside the movie details.

    - The output contains not only a full movie with a plot, genre, actors, directors, reviews and a poster, it also holds the details sent and received too and from the prompts to understand the interactions.

**Voila!** We just used Azure OpenAI generative models to make a movie from a general description of what the movie should entail and after passing the completions through various models it spits out a full film right down to the poster to go with it.  Check out the [poster](assets/examples/images/example.jpg) and [movie](assets/examples/json/example.json) assets to see some examples of results. For a full catalog, check out [Battlecabbage Media](https://battlecabbage-movies.azurewebsites.net)

### Learnings:

- Prompt engineering is critical!

    - There were plenty of situations where a single word made all the difference in the response we got. Being very specific about results was critical to how we could format the results to be returned (e.g. json) but also including key types of data that we wanted. Short, sweet and to the point.

- Models Matter!

    - Originally we started with GPT3.5 Turbo. While the response was generally faster the completions compared to GPT-4 were not nearly as descriptive or interesting. The JSON support was immediately better in later versions of 3.5 Turbo and GPT 4 natively.

- Micro requests help!

    - Originally we sent multiple steps like the review and image prompt request as part of the movie generation request.  While we received everything back everything felt short and limited. We had not nearly as much influence on the completions and was just generally speaking, too much at once. We could have bumped up the tokens to help but wanted to keep the window small.

- Tokens are a thing!

    - Like the need for breaking into smaller requests, the token count allowed for the completions helped minimize certain requests from going wild and allowed others to expand thoughts. An example is the image prompt would not only give us the prompt to generate the image but would go on a journey explaining why it provided the prompt, which we would quickly discard. Allow lots of tokens, it will use lots of tokens.

- Models like instructions!

    - Originally the prompt sent to a completion endpoint included the intent, data and request. It was hard to work with. Also made the prompt messy and often inconsistent results. Using a chat completion method, we gave the model explicit system instructions on its purpose and what it should expect to receive in the prompt. So what went from "Take this info, give us this response, do not do this and it has to be like this". Changed to "You are an assistant that will get information in a specific format of info:info and do this with it", "Here is info:info". The model seemed to be more creative as it was able to discern the actual data from the request better (purely observational but its grounding? the model)

- Ask AI!

    - Hours were spent trying to determine the structure of a poster, where a title should go, processing the image to determine the best contrast color of the title, etc, or just ask an AI model to handle that. Not only did we have to break down the generation into multiple smaller steps, we also introduced more logical steps. We found often we were having human interpretation of a common concept could be easily handled by the models and thus why so many steps came to be. Once we were clear on what we were trying to accomplish, AI models handled the rest.