# image_scraper.py
import requests
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

class IrishLandmarkScraper:
    def __init__(self):
        self.landmarks = {
            'cliffs_of_moher': ['Cliffs of Moher Ireland', 'Cliffs Moher tourist'],
            'giants_causeway': ['Giants Causeway Ireland', 'Giants Causeway stones'],
            'ring_of_kerry': ['Ring of Kerry Ireland', 'Kerry landscape Ireland'],
            'dublin_castle': ['Dublin Castle Ireland', 'Dublin Castle courtyard'],
            'killarney_national_park': ['Killarney National Park', 'Killarney lakes Ireland'],
            'rock_of_cashel': ['Rock of Cashel Ireland', 'Cashel cathedral Ireland']
        }
    
    def download_images(self, limit=200):
        """Download images for each landmark using Selenium"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        for landmark, search_terms in self.landmarks.items():
            print(f"Downloading images for {landmark}...")
            os.makedirs(f'data/raw/{landmark}', exist_ok=True)
            
            for term in search_terms:
                try:
                    # Search for images
                    driver.get(f'https://www.bing.com/images/search?q={term.replace(" ", "+")}')
                    time.sleep(2)
                    
                    # Find image elements
                    images = driver.find_elements(By.CLASS_NAME, 'mimg')
                    count = 0
                    
                    for img in images[:limit//len(search_terms)]:
                        try:
                            src = img.get_attribute('src')
                            if src and src.startswith('http'):
                                img_path = f'data/raw/{landmark}/{term}_{count}.jpg'
                                urllib.request.urlretrieve(src, img_path)
                                count += 1
                                time.sleep(0.5)  # Be respectful to servers
                        except Exception as e:
                            print(f"Error downloading image: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing {term}: {e}")
        
        driver.quit()
    
    def clean_dataset(self):
        """Clean the dataset by removing corrupt or invalid images"""
        from PIL import Image
        import os

        print("Cleaning dataset...")
        for landmark in self.landmarks.keys():
            folder_path = f'data/raw/{landmark}'
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Try to open the image to verify it's valid
                    with Image.open(file_path) as img:
                        img.verify()  # Verify it's a valid image
                except Exception as e:
                    print(f"Removing corrupt image {file_path}: {e}")
                    os.remove(file_path)

        print("Dataset cleaning complete")
    
# Usage
if __name__ == "__main__":
    scraper = IrishLandmarkScraper()
    scraper.download_images(limit=150)  # 150 images per landmark
    scraper.clean_dataset()