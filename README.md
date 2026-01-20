# Linear-Semester-1-Reflection


I implemented a computer program where the user can enter a word that the program will use to find the top-k most “similar” words to that word. This will compute the cosine similarities between the embedding vector(s) for that word’s token id(s) and the embedding vectors for all the token id’s. The top-k greatest entries in the resulting vector will correspond to the top-k most similar words.

Before running main.py run the following command:

    pip install torch tiktoken transformers

To run:

    python main.py