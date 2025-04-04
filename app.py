import os
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
import seamless_m4t_pb2
import seamless_m4t_pb2_grpc
from concurrent import futures
import grpc
import io 
# Set Hugging Face cache path
os.environ["HF_HOME"] = "/path/to/custom/huggingface_cache"  # Change this to your desired path

# Load the transcription model and processor
MODEL_NAME = "facebook/seamless-m4t-v2-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_NAME} on {device}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2Model.from_pretrained(MODEL_NAME).to(device)


class SeamlessM4TServicer(seamless_m4t_pb2_grpc.SeamlessM4TServiceServicer):
    def SpeechToText(self, request, context):
        """
        Handles a unary SpeechToText request, processes the audio, and returns a transcription.
        """
        try:
            # Load audio from the request directly into memory
            print("Processing received audio in memory...")
            audio_data = torch.frombuffer(request.audio, dtype=torch.float32)
            
            # Convert the buffer to a waveform tensor
            waveform, sampling_rate = torchaudio.load(io.BytesIO(request.audio), format="wav")
            print(f"Loaded audio: shape={waveform.shape}, sampling_rate={sampling_rate}")

            # Resample to 16 kHz if necessary
            if sampling_rate != 16000:
                print("Resampling audio to 16kHz...")
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)

            # Convert stereo to mono if necessary
            if waveform.shape[0] > 1:
                print("Converting audio to mono...")
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Prepare the input for the model
            inputs = processor(
                audios=waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)

            # Generate transcription
            output_tokens = model.generate(**inputs, tgt_lang=request.tgt_lang, generate_speech=False)
            transcribed_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            print(f"Transcription result: {transcribed_text}")

            # Return the response
            return seamless_m4t_pb2.SpeechToTextResponse(text=transcribed_text)

        except Exception as e:
            print(f"Error in SpeechToText: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return seamless_m4t_pb2.SpeechToTextResponse(text="Error during transcription.")


def serve():
    """
    Start the gRPC server and listen for client connections.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    seamless_m4t_pb2_grpc.add_SeamlessM4TServiceServicer_to_server(SeamlessM4TServicer(), server)
    server.add_insecure_port("[::]: 9090")
    print("Server is running on port 9090...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

