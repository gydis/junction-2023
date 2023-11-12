"""Text processing functions"""
import urllib
from typing import Dict, Generator, Optional
import string

from selenium.webdriver.remote.webdriver import WebDriver

from config import Config
from agent.llm_utils import create_chat_completion
import os
from md2pdf.core import md2pdf
import hashlib

CFG = Config()


def split_text(text: str, max_length: int = 24000) -> Generator[str, None, None]:
    """Split text into chunks of a maximum length

    Args:
        text (str): The text to split
        max_length (int, optional): The maximum length of each chunk. Defaults to 8192.

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: If the text is longer than the maximum length
    """
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def summarize_text(
    url: str, text: str, question: str, driver: Optional[WebDriver] = None
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    if not text:
        return "Error: No text to summarize"

    summaries = []
    chunks = list(split_text(text))
    scroll_ratio = 1 / len(chunks)

    print(f"Summarizing url: {url} with total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        if driver:
            scroll_to_percentage(driver, scroll_ratio * i)

        #memory_to_add = f"Source: {url}\n" f"Raw content part#{i + 1}: {chunk}"

        #MEMORY.add_documents([Document(page_content=memory_to_add)])

        messages = [create_message(chunk, question, chunk_flag=True)]

        summary = create_chat_completion(
            model=CFG.fast_llm_model,
            messages=messages,
            max_tokens=200,
            temperature=0.2,
            repetition_penalty=15,
            top_p=0.95,
            summ=True,
        )
        summaries.append(summary)
        #memory_to_add = f"Source: {url}\n" f"Content summary part#{i + 1}: {summary}"

        #MEMORY.add_documents([Document(page_content=memory_to_add)])

    combined_summary = "\n".join(summaries)
    os.makedirs(os.path.dirname(f"./outputs/chunk-summary/scrape-{hashlib.sha1(url.encode()).hexdigest()}.txt"), exist_ok=True)
    write_to_file(f"./outputs/chunk-summary/scrape-{hashlib.sha1(url.encode()).hexdigest()}.txt", url+combined_summary) # TESTING

    messages = [create_message(combined_summary, question, chunk_flag=True)]

    final_summary = create_chat_completion(
        model=CFG.fast_llm_model,
        messages=messages,
        max_tokens=400,
        temperature=0.2,
        repetition_penalty=20,
        summ=True,
    )
    print("Final summary length: ", len(combined_summary))
    # print(final_summary)

    return final_summary


def scroll_to_percentage(driver: WebDriver, ratio: float) -> None:
    """Scroll to a percentage of the page

    Args:
        driver (WebDriver): The webdriver to use
        ratio (float): The percentage to scroll to

    Raises:
        ValueError: If the ratio is not between 0 and 1
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Percentage should be between 0 and 1")
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {ratio});")


def create_message(chunk: str, question: str, chunk_flag: bool = False) -> Dict[str, str]:
    """Create a message for the chat completion

    Args:
        chunk (str): The chunk of text to summarize
        question (str): The question to answer

    Returns:
        Dict[str, str]: The message to send to the chat completion
    """
    return {
        "role": "user",
        "content": f'"""{chunk}"""\n'
        # note - question is actually a query, not the original question, nor a ? question
        # f'you must not exceed 100 word count.',
        f'using the above text, summarize it based on the following task or query: "{question}".\n' 
        f'if the query cannot be answered using the text, you must summarize the text in short.\n'
        f'include all factual information such as numbers, stats, quotes, etc if available.\n',
    } if not chunk_flag else {
        "role": "user",
        "content": f'"""{chunk}"""\n',
        # f'Here is the summary of the preceding text, including all factual information such as numbers, stats, quotes, etc if available.\n',
    }

def write_to_file(filename: str, text: str) -> None:
    """Write text to a file

    Args:
        text (str): The text to write
        filename (str): The filename to write to
    """
    print(f"Writing to file: {filename}") # TESTING
    with open(filename, "w") as file:
        file.write(text)

async def write_md_to_pdf(task: str, path: str, text: str) -> None:
    file_path = f"{path}/{task}"
    write_to_file(f"{file_path}.md", text)
    md_to_pdf(f"{file_path}.md", f"{file_path}.pdf")
    print(f"{task} written to {file_path}.pdf")

    encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")

    return encoded_file_path

def read_txt_files(directory):
    all_text = ''

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                all_text += file.read() + '\n'

    return all_text


def md_to_pdf(input_file, output_file):
    md2pdf(output_file,
           md_content=None,
           md_file_path=input_file,
           css_file_path=None,
           base_url=None)
