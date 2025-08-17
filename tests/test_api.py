import requests
import os

# Base URL for the API
BASE_URL = "https://tds-project-2-may-2025.onrender.com"

def test_health_check():
    """Tests the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_data():
    """Tests the main analyze_data endpoint with a file upload."""
    # Create a dummy questions.txt file
    with open("questionss.txt", "w") as f:
        f.write("Scrape the top 5 films from https://en.wikipedia.org/wiki/List_of_highest-grossing_films")

    with open("questionss.txt", "rb") as f:
        files = {"questions_txt": f}
        response = requests.post(f"{BASE_URL}/api/", files=files)
    
    # Clean up the dummy file
    os.remove("questionss.txt")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "completed"
    assert response_json["workflow_type"] == "multi_step_web_scraping"

if __name__ == "__main__":
    test_health_check()
    test_analyze_data()
    print("All tests passed!")