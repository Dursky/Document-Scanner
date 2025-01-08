import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class DocumentClassifier:
    def __init__(self, model_path=None):
        self.classes = [
            'advertisement', 'budget', 'email', 'file_folder', 'form',
            'handwritten', 'invoice', 'letter', 'memo', 'news_article',
            'presentation', 'questionnaire', 'resume', 'scientific_publication',
            'scientific_report', 'specification'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))
        return model.to(self.device)

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image):
        with torch.no_grad():
            preprocessed = self.preprocess_image(image)
            outputs = self.model(preprocessed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return self.classes[predicted.item()], confidence.item()

    def get_processing_params(self, doc_type):
        params = {
            'invoice': {'MORPH': 7, 'CANNY': 75, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 19},
            'letter': {'MORPH': 5, 'CANNY': 80, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 15},
            'handwritten': {'MORPH': 3, 'CANNY': 95, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 11},
            'scientific_publication': {'MORPH': 7, 'CANNY': 70, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 21},
            'form': {'MORPH': 5, 'CANNY': 85, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 17},
            'email': {'MORPH': 5, 'CANNY': 80, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 15},
            'resume': {'MORPH': 5, 'CANNY': 80, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 17},
            'specification': {'MORPH': 7, 'CANNY': 75, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 19},
            'scientific_report': {'MORPH': 7, 'CANNY': 70, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 21},
            'presentation': {'MORPH': 5, 'CANNY': 85, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 17},
            'advertisement': {'MORPH': 7, 'CANNY': 90, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 19},
            'budget': {'MORPH': 5, 'CANNY': 75, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 15},
            'memo': {'MORPH': 5, 'CANNY': 80, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 15},
            'news_article': {'MORPH': 7, 'CANNY': 75, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 19},
            'questionnaire': {'MORPH': 5, 'CANNY': 85, 'GAUSSIAN_BLUR': (3,3), 'ADAPTIVE_THRESH_BLOCK': 17},
            'file_folder': {'MORPH': 7, 'CANNY': 85, 'GAUSSIAN_BLUR': (5,5), 'ADAPTIVE_THRESH_BLOCK': 19}
        }
        return params.get(doc_type, params['form'])