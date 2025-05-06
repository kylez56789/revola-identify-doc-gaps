import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


def is_valid_url(url):
    """Check if the URL is valid and belongs to the same domain."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_base_domain(url):
    """Extract the base domain from a URL."""
    parsed_uri = urlparse(url)
    domain = "{uri.netloc}".format(uri=parsed_uri)
    return domain


def handle_cookie_popup(driver):
    try:
        # Wait for cookie banner and click accept
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "cookiebar-ok"))
        )
        accept_button.click()

        # Wait for banner removal from DOM
        WebDriverWait(driver, 5).until(
            EC.invisibility_of_element_located((By.ID, "cookie-banner"))
        )
    except TimeoutException:
        print("No cookie banner found or already dismissed")


def remove_cookie_elements(driver):
    driver.execute_script(
        """
        const selectors = [
            '#cookie-banner', 
            '.cookie-consent',
            'div[aria-label*="cookie"]'
        ];
        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        });
    """
    )


def find_links_on_page(driver, base_domain):
    """Extract and validate all links on a page using Selenium."""
    links = []

    # Find all anchor tags on the page
    anchor_tags = driver.find_elements(By.TAG_NAME, "a")

    for tag in anchor_tags:
        try:
            # Get the href attribute of the anchor tag
            href = tag.get_attribute("href")
            if not href:
                continue

            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(driver.current_url, href)

            # Parse the new URL
            parsed_url = urlparse(absolute_url)

            # Validate scheme and domain
            if parsed_url.scheme not in ("http", "https"):
                continue

            if parsed_url.netloc != base_domain:
                continue

            # Normalize URL to always use https
            normalized_url = parsed_url._replace(scheme="https").geturl()

            # Filter out non-html resources
            if any(
                parsed_url.path.endswith(ext)
                for ext in (".pdf", ".jpg", ".png", ".docx")
            ):
                continue

            # Normalize URL by removing fragments
            final_url = urlparse(normalized_url)._replace(fragment="").geturl()

            links.append(final_url)

        except Exception as e:
            print(f"Error processing link: {str(e)}")
            continue

    return list(set(links))  # Remove duplicates


def start_evaluation(company_url):
    """Main function to start the website evaluation process."""
    # Validate URL format
    if not is_valid_url(company_url):
        raise ValueError(
            "Invalid URL format. Please provide a complete URL including http:// or https://"
        )

    base_domain = get_base_domain(company_url)
    print(f"Starting evaluation for: {company_url}")
    print(f"Base domain: {base_domain}")

    # Initialize data structures
    visited_urls = set()
    pages_to_visit = [company_url]
    extracted_text = {}
    image_urls = []

    extracted_text, image_urls = crawl_website(
        base_domain, visited_urls, pages_to_visit, extracted_text, image_urls
    )
    print("Crawling completed.")
    return base_domain, visited_urls, pages_to_visit, extracted_text, image_urls


def crawl_website(
    base_domain, visited_urls, pages_to_visit, extracted_text, image_urls, max_pages=100
):
    """Crawl the website and extract text and images using Selenium."""
    options = Options()
    options.add_argument("--disable-notifications")  # Disable browser notifications

    count = 0
    driver = webdriver.Chrome(
        options=options
    )  # Ensure you have the ChromeDriver installed

    while pages_to_visit and count < max_pages:
        # Get the next URL to visit
        current_url = pages_to_visit.pop(0)

        # Normalize the current URL to always use https
        current_url = urlparse(current_url)._replace(scheme="https").geturl()

        print(f"Visiting: {current_url} (Page {count + 1}/{max_pages})")
        # Skip if already visited
        if current_url in visited_urls:
            continue

        # Mark as visited
        visited_urls.add(current_url)
        count += 1

        try:
            # Load the page in Selenium
            driver.get(current_url)

            # handle_cookie_popup(driver)
            # remove_cookie_elements(driver)

            # Check if the page source is valid
            if not driver.page_source:
                print(f"Failed to load page: {current_url}")
                continue

            # Extract text from the page
            soup = BeautifulSoup(driver.page_source, "html.parser")
            page_text = extract_text_from_page(soup)
            extracted_text[current_url] = page_text

            # Extract images from the page
            page_images = extract_images_from_page(driver, current_url)
            image_urls.extend(page_images)

            # Find links to other pages on the same domain
            links = find_links_on_page(driver, base_domain)

            # Add new links to pages_to_visit
            for link in links:
                if link not in visited_urls and link not in pages_to_visit:
                    pages_to_visit.append(link)

            # Be polite: add a small delay between requests
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {current_url}: {str(e)}")

    driver.quit()
    return extracted_text, image_urls


def extract_text_from_page(soup):
    """Extract all visible text from a page using Selenium."""
    # Check if the page source is valid
    if not soup:
        return ""

    # Remove non-visible elements like <script>, <style>, etc.
    for element in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "aside"]
    ):
        element.extract()

    # Remove cookie consent banners by class or ID
    for element in soup.find_all(class_="cookie-banner"):
        element.decompose()  # Completely remove the element
    for element in soup.find_all(id="cookie-consent"):
        element.decompose()

    # Extract visible text
    text = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    text = " ".join(text.split())
    # print(text)
    return text


def extract_images_from_page(driver, base_url):
    """Extract all image URLs from a page using Selenium."""
    image_urls = []

    # Find all image elements on the page
    image_elements = driver.find_elements(By.TAG_NAME, "img")

    for img in image_elements:
        # Get the 'src' attribute of the image
        img_url = img.get_attribute("src")
        if img_url:
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(base_url, img_url)
            image_urls.append(absolute_url)

    return image_urls
