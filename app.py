import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) #removing the last layer of the model that was FC for output
model = model.cuda()  # Move the model to GPU for faster computation
model.eval()

# Define image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).cuda()  # Move the image tensor to GPU

# Calculate image embeddings
def calculate_embedding(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze().cpu().numpy()  # Move the embedding back to CPU

# Compare embeddings using cosine similarity
def compare_embeddings(embedding1, embedding2):
    embedding1 = torch.tensor(embedding1).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    embedding2 = torch.tensor(embedding2).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    return nn.functional.cosine_similarity(embedding1, embedding2)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        images.append(img_path)
    return images


input_image_path = "image_1.png"
input_embedding = calculate_embedding(input_image_path)

database_folder = "database_folder"
database_images = load_images_from_folder(database_folder)

similarities = []
for image_path in database_images:
    database_embedding = calculate_embedding(image_path)
    similarity = compare_embeddings(input_embedding, database_embedding)
    similarities.append((image_path, similarity.item()))

sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

num_results_to_show = 3
for i in range(num_results_to_show):
    image_path, similarity = sorted_similarities[i]
    image = Image.open(image_path)
    image.show()
    print(f"Similarity with {image_path}: {similarity:.4f}")

