# Simple demo script to test background place detection
import os
from background_place_detector import BackgroundPlaceDetector

def demo():
    print("🎯 Background Place Detection Demo")
    print("="*50)
    
    # Initialize the detector
    print("Loading Places365 model...")
    detector = BackgroundPlaceDetector(arch='resnet18')
    
    # Test with the sample image (if it exists)
    test_image = '12.jpg'
    if os.path.exists(test_image):
        print(f"\n📸 Testing with sample image: {test_image}")
        detector.analyze_image(test_image)
    else:
        print(f"\n⚠️  Sample image {test_image} not found.")
        print("You can download a test image or use your own!")
    
    # Show available place categories
    print(f"\n📋 Available Place Categories ({len(detector.classes)}):")
    print("-" * 40)
    
    # Group some common places
    common_places = [
        'church/indoor', 'church/outdoor', 'amusement_park', 'cafeteria',
        'fastfood_restaurant', 'beach', 'forest/broadleaf', 'office',
        'kitchen', 'bedroom', 'library/indoor', 'museum/indoor'
    ]
    
    print("🏆 Common Background Places Detected:")
    for place in common_places:
        if place in detector.classes:
            print(f"  ✅ {place.replace('_', ' ').replace('/', ' - ').title()}")
    
    print(f"\n💡 Total place categories available: {len(detector.classes)}")
    print("   Including: restaurants, parks, churches, offices, beaches,")
    print("   forests, museums, libraries, bedrooms, kitchens, and many more!")

if __name__ == "__main__":
    demo()