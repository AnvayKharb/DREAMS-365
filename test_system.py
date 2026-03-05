# Test script to demonstrate background place detection capabilities
from background_place_detector import BackgroundPlaceDetector
import os

def test_capabilities():
    print("🎯 Background Place Detection System")
    print("="*60)
    
    # Initialize the detector
    print("📥 Initializing Places365 model...")
    detector = BackgroundPlaceDetector(arch='resnet18')
    
    print(f"\n✅ System Ready!")
    print(f"📊 Can detect {len(detector.classes)} different background places")
    
    # Show some example place categories
    print(f"\n🏷️  Examples of places your system can detect:")
    print("-" * 50)
    
    examples = {
        "Religious": ["abbey", "church/indoor", "church/outdoor"],
        "Recreation": ["amusement_park", "beach", "playground", "park"],
        "Food & Dining": ["cafeteria", "restaurant", "fastfood_restaurant", "bar", "diner"],
        "Nature": ["forest/broadleaf", "forest_path", "lake/natural", "mountain", "ocean"],
        "Work & Study": ["office", "library/indoor", "classroom", "laboratory_wet"],
        "Home": ["kitchen", "bedroom", "living_room", "bathroom", "dining_room"],
        "Culture": ["museum/indoor", "art_gallery", "theater/indoor", "concert_hall"],
        "Transportation": ["airport_terminal", "train_station/platform", "subway_station/platform"]
    }
    
    for category, places in examples.items():
        print(f"\n🏆 {category}:")
        for place in places:
            if place in detector.classes:
                print(f"   ✅ {place.replace('_', ' ').replace('/', ' - ').title()}")
    
    print(f"\n🎯 How to use:")
    print("1. Take any photo")
    print("2. Run: python background_place_detector.py --image your_photo.jpg")
    print("3. Get instant background place identification!")
    
    print(f"\n💡 Your system can tell you if a photo was taken in:")
    print("   • A church, mosque, or temple")
    print("   • A park, beach, or forest")
    print("   • A restaurant, cafe, or bar")
    print("   • An office, library, or classroom")
    print("   • And hundreds more specific places!")
    
    print(f"\n🚀 Ready to use! Just provide any image file.")

if __name__ == "__main__":
    test_capabilities()