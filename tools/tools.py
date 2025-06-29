import tempfile
import requests
import os

from time import sleep
from urllib.parse import urlparse
from typing import Optional, List
import yt_dlp
import imageio
from google.genai import types

from PIL import Image
from smolagents import CodeAgent, tool, OpenAIServerModel, LiteLLMModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

@tool
def use_vision_model(question: str, images: List[Image.Image]) -> str:
    """
    Use a Vision Model to answer a question about a set of images.  
    Always use this tool to ask questions about a set of images you have been provided.
    This function uses an image-to-text AI model.  
    You can ask a question about a list of one image or a list of multiple images.  
    So, if you have multiple images that you want to ask the same question of, pass the entire list of images to the model.
    Ensure your prompt is specific enough to retrieve the exact information you are looking for.
    
    Args:
        question: The question to ask about the images.  Type: str
        images: The list of images to as the question about.  Type: List[PIL.Image.Image]
    """
    image_model_name = "gemini/gemini-1.5-flash"
    
    print(f'Leveraging model {image_model_name}')
    image_model =LiteLLMModel(model_id=image_model_name, 
                           api_key=os.getenv("GEMINI_KEY"),
                           temperature=0.2
                           )

    content = [
        {
            "type": "text",
            "text": question
        }
    ]
    print(f"Asking model a question about {len(images)} images")
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
    print(f'Model returned: {output}')
    return output

@tool
def review_youtube_video(url: str, question: str) -> str:
    """
    Reviews a YouTube video and answers a specific question about that video.  

    Args:
        url (str): the URL to the YouTube video.  Should be like this format: https://www.youtube.com/watch?v=9hE5-98ZeCg
        question (str): The question you are asking about the video
    """
    try:
        client = genai.Client(api_key=os.getenv('GEMINI_KEY'))
        model = 'gemini-2.0-flash-lite'
        response = client.models.generate_content(
            model=model,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url)
                    ),
                    types.Part(text=question)
                ]
            )
        )
        return response.text
    except Exception as e:
        return f"Error asking {model} about video: {str(e)}"

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
def read_file(filepath: str ) -> str:
    """
    Used to read the content of a file.  Returns the content as a string.
    Will only work for text-based files, such as .txt files or code files.
    Do not use for audio or visual files. 
    
    Args:
        filepath (str): The path to the file to be read.

    Returns:
        str: Content of the file as a string.
    
    Raises:
        IOError: If there is an error opening or reading from the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        print(content)
        return content
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except IOError as e:
        print(f"Error reading file: {str(e)}")

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

import whisper

@tool
def youtube_transcribe(url: str) -> str:
    """
    Transcribes a YouTube video.  Use when you need to process the audio from a YouTube video into Text.

    Args:
        url: Url of the YouTube video
    """
    model_size: str = "small"
    # Load model
    model = whisper.load_model(model_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
            'quiet': True,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'force_ipv4': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        audio_path = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.wav')), None)
        if not audio_path:
            raise RuntimeError("Failed to find audio")

        # Transcribe
        result = model.transcribe(audio_path)
        return result['text']

@tool
def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file.  Use when you need to process audio data.
    DO NOT use this tool for YouTube video; use the youtube_transcribe tool to process audio data from YouTube.
    Use this tool when you have an audio file in .mp3, .wav, .aac, .ogg, .flac, .m4a, .alac or .wma

    Args:
        audio_file_path: Filepath to the audio file (str)
    """
    model_size: str = "small"
    # Load model
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file_path)
    return result['text']

# global driver
# driver = initialize_driver()