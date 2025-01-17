"""Selenium web scraping module."""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from sys import platform

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from fastapi import WebSocket
from langchain.document_loaders import PyMuPDFLoader
from langchain.retrievers import ArxivRetriever

import processing.text as summary

from config import Config
from processing.html import extract_hyperlinks, format_hyperlinks

from concurrent.futures import ThreadPoolExecutor

import os
from processing.text import write_to_file 
import hashlib # TESTING

executor = ThreadPoolExecutor()

FILE_DIR = Path(__file__).parent.parent
CFG = Config()


async def async_browse(url: str, question: str, websocket: WebSocket) -> str:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The search query which generated the link
        websocket (WebSocketManager): The websocket manager

    Returns:
        str: The answer and links to the user
    """
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=8)

    print(f"Scraping url {url} with question {question}")
    await websocket.send_json(
        {
            "type": "logs",
            "output": f"🔎 Browsing the {url} for relevant about: {question}...",
        }
    )

    try:
        driver, text = await loop.run_in_executor(
            executor, scrape_text_with_selenium, url
        )
        await loop.run_in_executor(executor, add_header, driver)
        summary_text = await loop.run_in_executor(
            executor, summary.summarize_text, url, text, question, driver
        )

        await websocket.send_json(
            {
                "type": "logs",
                "output": f"📝 Information gathered from url {url}: {summary_text}",
            }
        )

        return f"Information gathered from url {url}: {summary_text}"
    except Exception as e:
        print(f"An error occurred while processing the url {url}: {e}")
        return ""
        # return f"Error processing the url {url}: {e}"

def scrape_text_with_selenium(url: str) -> tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available = {
        "chrome": ChromeOptions,
        "safari": SafariOptions,
        "firefox": FirefoxOptions,
    }

    options = options_available[CFG.selenium_web_browser]()
    options.add_argument(f"user-agent={CFG.user_agent}")
    options.add_argument("--headless")
    options.add_argument("--enable-javascript")

    if CFG.selenium_web_browser == "firefox":
        driver = webdriver.Firefox(options=options)
    elif CFG.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = webdriver.Safari(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--no-sandbox")
        options.add_experimental_option("prefs", {"download_restrictions": 3})
        driver = webdriver.Chrome(options=options)

    print(f"scraping url {url}...")
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # check if url is a pdf or arxiv link
    if url.endswith(".pdf"):
        text = scrape_pdf_with_pymupdf(url)
    elif "arxiv" in url:
        # parse the document number from the url
        doc_num = url.split("/")[-1]
        text = scrape_pdf_with_arxiv(doc_num)
    else:
        # Get the HTML content directly from the browser's DOM
        page_source = driver.execute_script("return document.body.outerHTML;")
        soup = BeautifulSoup(page_source, "html.parser")

        for script in soup(["script", "style"]):
            script.extract()

        # text = soup.get_text()
        text = get_text(soup)

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # WTF is this? why double-space
    text = "\n".join(chunk for chunk in chunks if chunk)
    os.makedirs(os.path.dirname(f"./outputs/scrapes/scrape-{hashlib.sha1(url.encode()).hexdigest()}.txt"), exist_ok=True)
    write_to_file(f"./outputs/scrapes/scrape-{hashlib.sha1(url.encode()).hexdigest()}.txt", url+text) # TESTING

    return driver, text


def get_text(soup):
    """Get the text from the soup

    Args:
        soup (BeautifulSoup): The soup to get the text from

    Returns:
        str: The text from the soup
    """
    text = ""
    tags = ["h1", "h2", "h3", "h4", "h5", "p"]
    for element in soup.find_all(tags):  # Find all the <p> elements
        text += element.text + "\n\n"
    return text


def scrape_links_with_selenium(driver: WebDriver, url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): The webdriver to use to scrape the links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


def add_header(driver: WebDriver) -> None:
    """Add a header to the website

    Args:
        driver (WebDriver): The webdriver to use to add the header

    Returns:
        None
    """
    driver.execute_script(open(f"{FILE_DIR}/js/overlay.js", "r").read())


def scrape_pdf_with_pymupdf(url) -> str:
    """Scrape a pdf with pymupdf

    Args:
        url (str): The url of the pdf to scrape

    Returns:
        str: The text scraped from the pdf
    """
    loader = PyMuPDFLoader(url)
    doc = loader.load()
    return str(doc)


def scrape_pdf_with_arxiv(query) -> str:
    """Scrape a pdf with arxiv
    default document length of 70000 about ~15 pages or None for no limit

    Args:
        query (str): The query to search for

    Returns:
        str: The text scraped from the pdf
    """
    retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)
    docs = retriever.get_relevant_documents(query=query)
    return docs[0].page_content