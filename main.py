import torch # pytorch
import tiktoken # for the tokenizer
from transformers import GPT2LMHeadModel # basic GPT-2 Model

# gpt2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# user input

word = input("Enter a word (keep it simple!): ")

while (word != "exit"): 
    
    word = f" {word}" # gpt-2 words typically have a leading space

    # tokenize the word
    token_ids = enc.encode(word) 

    print(token_ids)

    token_ids = torch.tensor(token_ids)



    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # this is the word-token embedding matrix for gpt-2. Essentially just maps every token id to its corresponding embedding vector.
    wte = model.transformer.wte.weight # Dimensions: vocab_size x embedding dimension (768 for this model)


    token_embds = wte[token_ids] # dimensions: num_tokens x embedding_dim

    # a word can be composed of multiple tokens, so we take the mean of the embedding vectors over all the tokens to build a word embedding
    word_embedding = token_embds.mean(dim=0) # embd_dim sized vector

    # normalize the vector, makes it easier to compute cosine similarity if all magnitudes are 1
    word_embedding = word_embedding / word_embedding.norm()

    # compute the norm of each row of the wte
    wte_rows_norm = wte.norm(dim=1, keepdim=True)

    # normalize each row of the wte by dividing each element by the norm of the row its in
    wte = wte / wte_rows_norm

    # matrix multiplication
    cos_similarities = wte @ word_embedding # Multiplying a (vocab_size x embd_dim) matrix and a (embd_dim x 1) matrix yields a (vocab_size x 1) matrix

    # optional: mask the tokens of the user-inputted word so that they aren't in the top-k
    cos_similarities[token_ids] = -float("inf")


    k = 20
    # get top-k. indices will store the top-k token id's
    values, indices = torch.topk(cos_similarities, k)


    print(f"\nTop {k} Similar Tokens:")

    i = 0
    for idx in indices:
        print(f"{i+1}. '{enc.decode([idx.item()])}'")
        i+=1

    print("\nSome tokens don't actualy represent anything just by themselves, and so you might get something like �.")

    print("\nType exit to quit.")
    word = input("Enter a word (keep it simple!): ")



























