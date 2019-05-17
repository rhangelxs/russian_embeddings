# API server for word embeddings for Russian language

Docker images for Russian various word2vec models. Now, there is only one model ([Araneum](https://rusvectores.org/en/models/)) which is FastText model created with [Gensim](github.com/RaRe-Technologies/gensim) package. The model got from RusVectores

Builded images are stored at DockerHub here: [https://hub.docker.com/r/rhangelxs/russian_embeddings](hub.docker.com/r/rhangelxs/russian_embeddings). Individual models marked with a tag. Anareum model DockerHub's tag is: `rhangelxs/russian_embeddings:araneum_none_fasttextcbow_300_5_2018`

GitHub: https://github.com/rhangelxs/russian_embeddings


## Idea

Docker images build with Flask, Arrested. By defaults flask development server is used, but can be served by Gunicorn as well.

All individual models can have own `requirements.txt` and `Dockerfile`, check `araneum_none_fasttextcbow_300_5_2018` as an example.

## API

Default port is `8080`.

### Transform word to vector

Individual words can be sended with GET or POST requests. Below `{word}` is a placeholder that should be replaced with a word of interest.

* GET endpoint: /api/araneum_none_fasttextcbow_300_5_2018/v1/inference/{word}
  
  Example: /api/araneum_none_fasttextcbow_300_5_2018/v1/inference/тест

* POST endpoint: /api/araneum_none_fasttextcbow_300_5_2018/v1/inference

  Payload is:
  ```{"token": "{word}"}```
  
  Example od POST payload:
  ```{"token": "тест"}```
  
You can send multiple words to `POST endpoint` and in a responce you will get a mean vector of standard size.

### Calculate Word Mover's Distance

Additional available method is Word Mover's Distance, check Gensim documentation.

* POST endpoint: /api/araneum_none_fasttextcbow_300_5_2018/v1/wmdsimilarity
  
  Payload:
  ```json
  {
    "corpus": [
        [
            "слово",
            "второе",
            "третье"
        ], 
        [
            "здравствуйте"
        ],
        [
            "пока"
        ]
    ],
    "query": [
        "привет"
    ],
    "num_best": 5
  }
  ```
  
  The responce will have a list of ids of best matched indexes limited by `num_best` count.


### Credits

1. Kutuzov, A., & Kuzmenko, E. (2016, April). WebVectors: a toolkit for building web interfaces for vector semantic models. In *International Conference on Analysis of Images, Social Networks and Texts* (pp. 155-161). Springer, Cham.
