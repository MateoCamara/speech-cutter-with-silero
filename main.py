import os
import librosa
import soundfile as sf
from silero_vad import get_speech_timestamps, VADIterator
import wandb
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import torch

# Inicializar wandb
wandb.init(project="audio_segmentation", name="silero_vad_audio_cutting")

# Parámetros configurables
config = {
    "min_audio_length": 2.0,  # En segundos
    "speech_prob_threshold": 0.5,
    "window_size": 10,  # En milisegundos
    "target_sample_rate": 16000  # Frecuencia de muestreo fija
}
wandb.config.update(config)

# Ruta de entrada y salida
input_base_path = "../cut_speech_data"
output_base_path = "../cut_speech_results"

# Modelo Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


# Función para procesar un archivo con resampling
def process_audio_file(file_path, output_dir):
    # Cargar audio
    audio, sample_rate = torchaudio.load(file_path)

    # Convertir a mono
    audio = audio.mean(dim=0, keepdim=True)

    # Resamplear si es necesario
    if sample_rate != config["target_sample_rate"]:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=config["target_sample_rate"])
        audio = resampler(audio)
        sample_rate = config["target_sample_rate"]

    # Detectar voz con Silero VAD
    timestamps = get_speech_timestamps(audio, model,
                                       sampling_rate=sample_rate,
                                       threshold=config["speech_prob_threshold"],
                                       min_silence_duration_ms=config["window_size"] * 2)

    # Filtrar fragmentos cortos
    filtered_timestamps = [
        t for t in timestamps if (t['end'] - t['start']) / sample_rate >= config["min_audio_length"]
    ]

    # Guardar los fragmentos
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    for idx, t in enumerate(filtered_timestamps):
        fragment = audio[:, t['start']:t['end']]
        output_file = os.path.join(output_dir, f"{base_name}_{idx}.wav")
        torchaudio.save(output_file, fragment, sample_rate)

    # Datos para visualización
    return len(filtered_timestamps), [(t['end'] - t['start']) / sample_rate for t in filtered_timestamps]


# Procesar todos los archivos en la estructura
file_stats = []
for speaker in os.listdir(input_base_path):
    speaker_path = os.path.join(input_base_path, speaker)
    if os.path.isdir(speaker_path):  # Procesar solo carpetas de nivel superior
        for book_folder in os.listdir(speaker_path):
            book_path = os.path.join(speaker_path, book_folder)
            book_name = book_folder
            if os.path.isdir(book_path):
                # Crear ruta de salida equivalente
                output_folder = os.path.join(output_base_path, speaker, book_folder)
                os.makedirs(output_folder, exist_ok=True)

                # Obtener todos los archivos MP3
                audio_files = [f for f in os.listdir(book_path) if f.endswith(".mp3")]
                for file in tqdm(audio_files, desc=f"Procesando {speaker}/{book_folder}"):
                    file_path = os.path.join(book_path, file)
                    wav_file_path = file_path.replace(".mp3", ".wav")

                    # Convertir a WAV si es necesario
                    if not os.path.exists(wav_file_path):
                        y, sr = librosa.load(file_path, sr=None)
                        if sr != config["target_sample_rate"]:
                            y = librosa.resample(y, orig_sr=sr, target_sr=config["target_sample_rate"])
                        sf.write(wav_file_path, y, config["target_sample_rate"])

                    # Procesar archivo de audio
                    num_fragments, fragment_lengths = process_audio_file(wav_file_path, output_folder)
                    file_stats.append(
                        {"file": file, "num_fragments": num_fragments, "fragment_lengths": fragment_lengths})

                    # Log en wandb
                    wandb.log({f"{book_name}_num_fragments": num_fragments,
                               f"{book_name}_fragment_lengths": wandb.Histogram(fragment_lengths)})

# Resumen en wandb
total_fragments = sum(stat["num_fragments"] for stat in file_stats)
all_lengths = [length for stat in file_stats for length in stat["fragment_lengths"]]

wandb.log({
    "total_fragments": total_fragments,
    "all_fragment_lengths": wandb.Histogram(all_lengths),
    "mean_fragment_length": sum(all_lengths) / len(all_lengths)
})

wandb.finish()
