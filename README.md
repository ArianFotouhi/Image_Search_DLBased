# Image Search

In this app, we have developed an image similarity search engine utilizing the ResNet-50 model for content-based image retrieval. The objective is to find visually similar images from a database based on a given input image.

The ResNet-50 model is employed for feature extraction. We specifically utilize the early layers of the model to extract image features while excluding the final layer (classification layer). This process enables us to obtain meaningful image embeddings.

The core steps of the image similarity search are as follows:

Image Feature Extraction: The calculate_embedding() function is responsible for transforming an input image into a feature embedding. This involves preprocessing the image, resizing it to (224, 224), converting it to a tensor, and normalizing it. The pre-trained ResNet-50 model is then employed to generate the image's embedding, which is used for comparison.

Cosine Similarity Calculation: To compare the feature embeddings, we compute the cosine similarity between the embedding of the target image and the embeddings of all the other images in the database. Cosine similarity is a metric used to measure the similarity between two vectors in a multi-dimensional space, with values ranging from -1 to 1.

Sorting and Top Results: The computed similarity scores are stored in a list. We then sort this list in descending order to identify the most visually similar images to the target image. The top results, based on the highest similarity scores, are displayed prominently.

The app demonstrates an effective approach to content-based image retrieval by leveraging the power of deep learning and the ResNet-50 model. The utilization of cosine similarity enables efficient image comparison, facilitating the creation of a robust and user-friendly image similarity search engine.

Lastly, in considering dependencies install the required libraries as below:
pip install torch torchvision pillow
