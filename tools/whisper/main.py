from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import whisper
import os
import tempfile

app = FastAPI()

# Load the Whisper model
model = whisper.load_model("base")

@app.post("/v1/transcribe")
async def transcribe_audio(
    audio_file: UploadFile,
    language: str = Form("en")
):
    """
    Transcribe the uploaded audio file to text.
    """
    try:
        # Save the uploaded audio file to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, audio_file.filename)

        with open(temp_audio_path, "wb") as temp_audio_file:
            temp_audio_file.write(await audio_file.read())

        # Convert audio to the desired format using pydub
        audio = AudioSegment.from_file(temp_audio_path)
        converted_audio_path = os.path.join(temp_dir, "converted_audio.wav")
        audio.export(converted_audio_path, format="wav")

        # Transcribe the audio using Whisper
        transcription = model.transcribe(converted_audio_path, language=language)

        # Clean up temporary files
        os.remove(temp_audio_path)
        os.remove(converted_audio_path)
        os.rmdir(temp_dir)

        return JSONResponse(content={"transcription": transcription["text"]}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the app with uvicorn
# Command: uvicorn filename:app --reload
