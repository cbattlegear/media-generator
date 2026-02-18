# Battlecabbage Media Generator

This respository is host to a media generation tool that is part of a larger [Battlecabbage Media](https://github.com/Battlecabbage-Media) project. This tools makes use of multiple Azure OpenAI models.

- Azure Open AI

    - GPT-4

    - GPT-4 Turbo for Vision capabilities

    - Dalle3

The result is complete movie with title, tagline, description, content rating, critic review and more. In addition a poster is created to match the movie allowing for what is effectively an automatic movie maker. All of this data is largely utilized for [Battlecabbage Media](https://battlecabbage-movies.azurewebsites.net)

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