"""
Simple test script for the Fake News Detection API
Run this after starting the backend server to test if it's working correctly.
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    print("Testing GET /...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_health():
    """Test the health check endpoint"""
    print("Testing GET /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict(text):
    """Test the predict endpoint with sample text"""
    print(f"Testing POST /predict...")
    print(f"Input text: {text[:100]}...\n")
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n✅ Truth Probability: {result['truth_probability'] * 100:.2f}%")
        print(f"✅ Label: {result['label'].upper()}\n")
    else:
        print(f"Error: {response.text}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Fake News Detection API Test")
    print("=" * 60 + "\n")
    
    try:
        # Test basic endpoints
        test_root()
        test_health()
        
        # Test with sample true news
        print("-" * 60)
        print("Test 1: Likely TRUE news")
        print("-" * 60)
        true_news = """
        NASA's Perseverance rover successfully landed on Mars on February 18, 2021.
        The rover is equipped with scientific instruments to search for signs of ancient
        microbial life and collect rock samples for future return to Earth.
        """
        test_predict(true_news)
        
        # Test with sample fake news
        print("-" * 60)
        print("Test 2: Likely FAKE news")
        print("-" * 60)
        fake_news = """
        Scientists have discovered that drinking coffee makes you invisible for 3 hours.
        The government has been hiding this information for decades. A secret study
        documented over 10,000 cases of people becoming transparent after consuming caffeine.
        """
        test_predict(fake_news)
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
