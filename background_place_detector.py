# Background Place Detector using Places365
# Enhanced version for identifying background places like church, park, cafe, etc.

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import argparse
import glob

class BackgroundPlaceDetector:
    def __init__(self, arch='resnet18'):
        self.arch = arch
        self.model = None
        self.classes = None
        self.transform = None
        self._load_model()
        self._load_classes()
        self._setup_transform()
    
    def _load_model(self):
        """Load the pre-trained Places365 model"""
        model_file = f'{self.arch}_places365.pth.tar'
        if not os.access(model_file, os.R_OK):
            print(f"Downloading {model_file}...")
            weight_url = f'http://places2.csail.mit.edu/models_places365/{model_file}'
            os.system(f'curl -O {weight_url}')
        
        self.model = models.__dict__[self.arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ Model loaded: {self.arch}")
    
    def _load_classes(self):
        """Load the class labels"""
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.R_OK):
            print("Downloading categories...")
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system(f'curl -O {synset_url}')
        
        classes = []
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(classes)
        print(f"✅ Loaded {len(self.classes)} place categories")
    
    def _setup_transform(self):
        """Setup image preprocessing"""
        self.transform = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_place(self, image_path, top_k=5):
        """
        Predict the background place of an image
        
        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (probability, place_name) tuples
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            input_img = V(self.transform(img).unsqueeze(0))
            
            # Forward pass
            with torch.no_grad():
                logit = self.model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)
            
            # Get top predictions
            predictions = []
            for i in range(min(top_k, len(probs))):
                prob = float(probs[i])
                place = self.classes[idx[i]]
                predictions.append((prob, place))
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_image(self, image_path, detailed=True):
        """Analyze an image and print results"""
        print(f"\n🔍 Analyzing: {image_path}")
        predictions = self.predict_place(image_path)
        
        if not predictions:
            return
        
        print(f"📍 Background Place Detection Results:")
        print("-" * 50)
        
        for i, (prob, place) in enumerate(predictions, 1):
            confidence = prob * 100
            print(f"{i}. {place.replace('_', ' ').title()}")
            print(f"   Confidence: {confidence:.1f}%")
            
            if detailed and i == 1:
                # Provide context for the top prediction
                self._provide_context(place, confidence)
        
        return predictions[0] if predictions else None
    
    def _provide_context(self, place, confidence):
        """Provide additional context about the predicted place"""
        place_lower = place.lower()
        
        context_messages = {
            'church': "🏛️  Religious building detected",
            'park': "🌳  Outdoor recreational area detected", 
            'cafeteria': "☕  Dining/eating establishment detected",
            'restaurant': "🍽️  Food service establishment detected",
            'beach': "🏖️  Coastal/waterfront location detected",
            'forest': "🌲  Natural woodland area detected",
            'office': "🏢  Professional workspace detected",
            'kitchen': "👩‍🍳  Food preparation area detected",
            'library': "📚  Educational/reading space detected",
            'museum': "🏛️  Cultural/exhibition space detected",
            'bedroom': "🛏️  Private living space detected",
            'living_room': "🛋️  Social living space detected"
        }
        
        for key, message in context_messages.items():
            if key in place_lower:
                print(f"   {message}")
                break
        
        if confidence > 80:
            print(f"   ✅ Very confident prediction")
        elif confidence > 60:
            print(f"   ⚠️  Moderately confident prediction")
        else:
            print(f"   ❓ Low confidence - consider multiple possibilities")
    
    def batch_analyze(self, image_folder, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp']):
        """Analyze multiple images in a folder"""
        results = {}
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        print(f"\n📁 Found {len(image_files)} images in {image_folder}")
        
        for image_path in image_files:
            result = self.analyze_image(image_path, detailed=False)
            if result:
                results[image_path] = result
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Background Place Detector')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--arch', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'resnet152'],
                       help='Model architecture')
    parser.add_argument('--top-k', type=int, default=5, 
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BackgroundPlaceDetector(arch=args.arch)
    
    if args.image:
        # Analyze single image
        detector.analyze_image(args.image)
    elif args.folder:
        # Analyze folder of images
        detector.batch_analyze(args.folder)
    else:
        # Interactive mode
        print("\n🎯 Background Place Detector Ready!")
        print("Enter image path (or 'quit' to exit):")
        
        while True:
            image_path = input("\n📸 Image path: ").strip()
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
            if os.path.exists(image_path):
                detector.analyze_image(image_path)
            else:
                print("❌ File not found!")

if __name__ == "__main__":
    main()