import tempfile
import requests
import os
import numpy as np
import helium
import base64

from io import BytesIO
from time import sleep
from urllib.parse import urlparse
from typing import Optional, List
import yt_dlp
import imageio


from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, tool, OpenAIServerModel
from smolagents.agents import ActionStep



def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return

def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)

@tool
def use_vision_model(question: str, images: List[Image.Image]) -> str:
    """
    Use a Vision Model to answer a question about a set of images.  
    Always use this tool to ask questions about a set of images you have been provided.
    You can add a list of one image or multiple images.
    
    Args:
        question: The question to ask about the images.  Type: str
        images: The list of images to as the question about.  Type: List[PIL.Image.Image]
    """
    image_model_name = 'llama3.2-vision'
    image_model = OpenAIServerModel(
        model_id=image_model_name,
        api_base='http://localhost:11434/v1/',
        api_key='ollama',
        flatten_messages_as_text=False
    )

    image_model_name = 'llama3.2-vision'
    image_model = OpenAIServerModel(
        model_id=image_model_name,
        api_base='http://localhost:11434/v1/',
        api_key='ollama',
        flatten_messages_as_text=False
    )

    content = [
        {
            "type": "text",
            "text": question
        }
    ]

    for image in images:
        content.append({
            "type": "image",
            "image": image  # ✅ Directly the PIL Image, no wrapping
        })

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    output = image_model(messages).content
    return output

@tool
def youtube_frames_to_images(url: str, sample_interval_seconds: int = 5) -> List[Image.Image]:
    """
    Reviews a YouTube video and returns a List of PIL Images (List[PIL.Image.Image]), which can then be reviewed by a vision model.
    Only use this tool if you have been given a YouTube video that you need to analyze.
    This will generate a list of images, and you can use the use_vision_model tool to analyze those images
    Args:
        url: The Youtube URL
        sample_interval_seconds: The sampling interval (default is 5 seconds)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the video locally
        ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
            'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
            'quiet': True,
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'force_ipv4': True,  # Avoid IPv6 issues
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        # Find the downloaded file
        video_path = None
        for file in os.listdir(tmpdir):
            if file.endswith('.mp4'):
                video_path = os.path.join(tmpdir, file)
                break
        
        if not video_path:
            raise RuntimeError("Failed to download video as mp4")

        # ✅ Fix: Use `imageio.get_reader()` instead of `imopen()`
        reader = imageio.get_reader(video_path)  # Works for frame-by-frame iteration
        metadata = reader.get_meta_data()
        fps = metadata.get('fps')
        
        if fps is None:
            reader.close()
            raise RuntimeError("Unable to determine FPS from video metadata")

        frame_interval = int(fps * sample_interval_seconds)
        images: List[Image.Image] = []

        # ✅ Iterate over frames using `get_reader()`
        for idx, frame in enumerate(reader):
            if idx % frame_interval == 0:
                images.append(Image.fromarray(frame))

        reader.close()
        return images

@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Used when navigating web pages using Helium.  Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a temporary file and return the path.
    Useful for processing files from the GAIA API.
    
    Args:
        content: The content to save to the file
        filename: Optional filename, will generate a random name if not provided
        
    Returns:
        Path to the saved file
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    
    # Write content to the file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return f"File saved to {filepath}. You can read this file to process its contents."

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.  
    Use this tool when you are asked a question and told that there is a file or image provided.

    
    Args:
        url: The URL to download from
        filename: Optional filename, will generate one based on URL if not provided
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Parse URL to get filename if not provided
        print(f"Downloading file from {url}")
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                # Generate a random name if we couldn't extract one
                import uuid
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"File downloaded to {filepath}. You can now process this file."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract (if available).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text or error message
    """
    try:
        # Try to import pytesseract
        import pytesseract
        from PIL import Image
        
        # Open the image
        image = Image.open(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        return f"Extracted text from image:\n\n{text}"
    except ImportError:
        return "Error: pytesseract is not installed. Please install it with 'pip install pytesseract' and ensure Tesseract OCR is installed on your system."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.  
    To use this file you need to have saved it in a location and pass that location to the function.
    The download_file_from_url tool will save it by name to tempfile.gettempdir()
    
    Args:
        file_path: Path to the CSV file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    To use this file you need to have saved it in a location and pass that location to the function.
    The download_file_from_url tool will save it by name to tempfile.gettempdir()
    
    Args:
        file_path: Path to the Excel file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Run various analyses based on the query
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

global driver
driver = initialize_driver()