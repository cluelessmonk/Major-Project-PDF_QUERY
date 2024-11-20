from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time


def get_webscrape_data(user_query):
    """
    Automatically installs ChromeDriver, performs a Google search for the user query,
    retrieves the top 5 blog-like website links (excluding YouTube), ensures presence of <p> tags,
    extracts their content, and returns it in a concatenated format with domain names.

    Args:
        user_query (str): The query to search on Google.

    Returns:
        str: Concatenated content of the top 5 valid websites in the specified format.
    """
    # Configure WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run browser in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')

    # Use Service to manage ChromeDriver path
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Open Google
        driver.get("https://www.google.com")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )

        # Perform Google search
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(user_query)
        search_box.send_keys(Keys.RETURN)

        # Wait for search results to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div#search a"))
        )

        # Get top result links and filter blogs
        result_links = []
        results = driver.find_elements(By.CSS_SELECTOR, "div#search a")
        for result in results:
            href = result.get_attribute("href")
            if href and "http" in href and "youtube.com" not in href:
                result_links.append(href)

            # Stop after collecting a large number of candidate links (e.g., 10-15)
            if len(result_links) >= 10:
                break

        # Extract content from each link
        website_content = []
        for link in result_links:
            driver.get(link)
            time.sleep(3)  # Allow the website to load

            # Get HTML content
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")

            # Check for <p> tags and filter out irrelevant pages
            paragraphs = soup.find_all('p')  # Find all <p> tags
            if len(paragraphs) < 5:  # Heuristic: Skip pages with fewer than 5 <p> tags
                continue

            # Extract and concatenate text content
            text = ' '.join(
                p.get_text(separator=" ", strip=True) for p in paragraphs
            )  # Join all text from <p> tags

            # Skip pages with very little text content
            if len(text) < 300:  # Heuristic: Minimum content length
                continue

            # Extract domain name for formatting
            parsed_url = urlparse(link)
            domain = parsed_url.netloc  # Extracts the domain, e.g., 'www.wikipedia.com'

            # Add to formatted output
            website_content.append(f"{domain} --> {text}")

            # Stop after retrieving content from the top 5 valid links
            if len(website_content) >= 5:
                break

    except TimeoutException as e:
        print(f"TimeoutException: {str(e)}")
        return "Error: Failed to load content from one or more websites."

    finally:
        # Close the driver
        driver.quit()
    user_query_data = f"User query is --> {user_query} and the content is --> "
    # Concatenate all website content into a single string
    answer = ' , '.join(website_content)
    answer =  user_query_data +answer
    return answer

