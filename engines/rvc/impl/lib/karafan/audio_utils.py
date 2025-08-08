
#   MIT License - Copyright (c) 2023 - Captain FLAM & Jarredou
#
#   https://github.com/Captain-FLAM/KaraFan

import os, librosa, subprocess, tempfile, soundfile as sf, numpy as np
from scipy import signal

def Load_Audio(file, sample_rate, ffmpeg = None, output_path = None):

	
	# Load the audio file from User's input
	try:
		print("Load the audio file from User's input")
		audio, sr = sf.read(file, dtype='float32')

	# Corrupted file : try to correct it with ffmpeg
	except RuntimeError as e:
		print('<div style="font-size:18px;color:#ff0040;"><b>Your audio file is Bad encoded ! Trying to correct ...</b></div>')
		try:
			corrected = os.path.join(output_path, "X - CORRECTED.flac")

			if not os.path.exists(corrected):
				subprocess.run(f'"{ffmpeg}" -y -i "{file}" -codec:a flac -compression_level 5 -ch_mode mid_side -lpc_type cholesky -lpc_passes 1 -exact_rice_parameters 1 "{corrected}"', shell=True, text=True, capture_output=True, check=True)

			print("Load corrected audio file")
			audio, sr = sf.read(corrected, dtype='float32')
			
		except RuntimeError as e:
			print(f"Error : {e}");  return None, 0
		except Exception as e:
			print(f"Error : {e}");  return None, 0
	except Exception as e:
		print(f"Error : {e}");  return None, 0
	
	audio = audio.T

	# Convert mono to stereo (if needed)
	if audio.ndim == 1:
		audio = np.asfortranarray([audio, audio])

	# Convert to 44100 Hz if needed (for MDX models)
	if sr != sample_rate:
		audio = librosa.resample(audio, orig_sr = sr, target_sr = sample_rate, res_type = 'kaiser_best', axis=-1, fix=True)

	return audio, sample_rate

def Save_Audio(file_path, audio, sample_rate, output_format, cut_off, ffmpeg):

	# if output_format == 'PCM_16' or output_format == 'FLOAT':
	if output_format == 'WAV':
		sf.write(file_path + '.wav',  audio.T, sample_rate, format='wav', subtype = "PCM_16" if "int" in audio.dtype.name else "FLOAT")
	else:
		# Create a temporary file
		temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

		if output_format == 'FLAC':
			sf.write(temp, audio.T, sample_rate, format='wav', subtype='FLOAT')
			ffmpeg = f'"{ffmpeg}" -y -i "{temp.name}" -codec:a flac -compression_level 5 -ch_mode mid_side -frame_size {sample_rate} -lpc_type cholesky -lpc_passes 1 -exact_rice_parameters 1 "{file_path}.flac"'

		elif output_format == 'MP3':
			# TODO : Correct the BUG of Lame encoder which modify the length of audio results (~ +30 ms on short song, -30 ms on long song) ?!?!

			# about VBR/CBR/ABR		: https://trac.ffmpeg.org/wiki/Encode/MP3
			# about ffmpeg wrapper	: http://ffmpeg.org/ffmpeg-codecs.html#libmp3lame-1
			# recommended settings	: https://wiki.hydrogenaud.io/index.php?title=LAME#Recommended_encoder_settings

			# 320k is mandatory, else there is a weird cutoff @ 16khz with VBR parameters = ['-q','0'] !!
			# (equivalent to lame "-V0" - 220-260 kbps , 245 kbps average)
			# And also, parameters = ['-joint_stereo', '0'] (Separated stereo channels)
			# is WORSE than "Joint Stereo" for High Frequencies !
			# So let's use it by default for MP3 encoding !!

			sf.write(temp, audio.T, sample_rate, format='wav', subtype='PCM_16')
			ffmpeg = f'"{ffmpeg}" -y -i "{temp.name}" -codec:a libmp3lame -b:a 320k -q:a 0 -joint_stereo 1 -cutoff {cut_off} "{file_path}.mp3"'

		try:
			subprocess.run(ffmpeg, shell=True, text=True, capture_output=True, check=True)

		except subprocess.CalledProcessError as e:
			if e.returncode == 127:
				print('WARNING : "\nFFmpeg" is not installed on your system !!\n')
			else:
				print("Error :\n" + e.stderr + "\n" + e.stdout)

		temp.close()
		os.remove(temp.name)

def Normalize(audio, threshold_dB = -1.0):
	"""
	Normalize audio to -1.0 dB peak amplitude
	This is mandatory for SOME audio files because every process is based on RMS dB levels.
	(Models processing, Volumes Compensations & audio Substractions)
	"""
	audio = audio.T
	
	# Suppress DC shift (center on 0.0 vertically)
	audio -= np.mean(audio)

	# Normalize audio peak amplitude to -1.0 dB
	max_peak = np.max(np.abs(audio))
	if max_peak > 0.0:
		max_db = 10 ** (threshold_dB / 20)  # Convert -X dB to linear scale
		audio /= max_peak
		audio *= max_db

	return audio.T

def Silent(audio_in, sample_rate, threshold_dB = -50):
	"""
	Make silent the parts of audio where dynamic range (RMS) goes below threshold.
	Don't misundertand : this function is NOT a noise reduction !
	Its behavior is to clean the audio from "silent parts" (below -XX dB) to :
	- avoid the MLM model to work on "silent parts", and save GPU time
	- avoid the MLM model to produce artifacts on "silent parts"
	- clean the final audio files from residues of "silent parts"
	"""

	min_size		= int(1.000 * sample_rate)  # 1000 ms
	window_frame	= int(0.500 * sample_rate)  #  500 ms
	fade_duration	= int(0.300 * sample_rate)  #  450 ms
	fade_out		= np.linspace(1.0, 0.0, fade_duration)
	fade_in			= np.linspace(0.0, 1.0, fade_duration)

	start = 0; end = 0
	audio = audio_in.copy()
	audio_length = audio_in.shape[-1]

	for i in range(0, audio_length, window_frame):
		
		# TODO : Maybe use S=audio (Spectrogram) instead of y=audio ??
		RMS = np.max(librosa.amplitude_to_db(librosa.feature.rms(y=audio[:, i:(i + window_frame)], frame_length=window_frame, hop_length=window_frame)))
		if RMS < threshold_dB:
			end = i + window_frame
			# Last part (in case of silence at the end)
			if i >= audio_length - window_frame:
				if end - start > min_size:
					# Fade out
					if start > fade_duration:
						audio[:, start:(start + fade_duration)] *= fade_out
						start += fade_duration

					# Clean last part
					audio[:, start:audio_length] = 0.0
					break
		else:
			# Clean the "min_size" samples found
			if end - start > min_size:

				# print(f"RMS : {RMS} / start : {start} / end : {end}")

				# Fade out
				if start > fade_duration:
					audio[:, start:(start + fade_duration)] *= fade_out
					start += fade_duration

				# Fade in
				if end < audio_length - fade_duration:
					audio[:, (end - fade_duration):end] *= fade_in
					end -= fade_duration
		
				# Clean in between
				audio[:, start:end] = 0.0

			start = i
			
	return audio

# Linkwitz-Riley filter
#
# Avec cutoff = 17.4khz & -80dB d'atténuation:
#
# ordre =  4 => filtre target freq = 10500hz
# ordre =  6 => filtre target freq = 13200hz
# ordre =  8 => filtre target freq = 14300hz
# ordre = 10 => filtre target freq = 15000hz
# ordre = 12 => filtre target freq = 15500hz
# ordre = 14 => filtre target freq = 15800hz
# ordre = 16 => filtre target freq = 16100hz
#
# Avec cutoff = 17.4khz & -60dB d'atténuation:
#
# ordre =  4 => filtre target freq = 12500hz (-4900)
# ordre =  6 => filtre target freq = 14400hz
# ordre =  8 => filtre target freq = 15200hz (-2200)
# ordre = 10 => filtre target freq = 15700hz
# ordre = 12 => filtre target freq = 16000hz (-1640)
# ordre = 14 => filtre target freq = 16200hz
# ordre = 16 => filtre target freq = 16400hz

def Linkwitz_Riley_filter(type, cutoff, audio, sample_rate, order=8):
	
	# cutoff -= 2200

	nyquist = 0.5 * sample_rate
	normal_cutoff = cutoff / nyquist

	sos = signal.butter(order // 2, normal_cutoff, btype=type, analog=False, output='sos')
	filtered_audio = signal.sosfiltfilt(sos, audio, padlen=0, axis=1)

	return filtered_audio

# Band Pass filter
#
# Vocals -> lowest : 85 - 100 Hz, highest : 20 KHz
# Music  -> lowest : 30 -  50 Hz, highest : 18-20 KHz
#
# Voix masculine :
#
# Minimale : 85 Hz
# Fondamentale : 180 Hz
# Maximale (y compris les harmoniques) : 14 kHz
#
# Voix féminine :
#
# Minimale : 165 Hz
# Fondamentale : 255 Hz
# Maximale (y compris les harmoniques) : 16 kHz
#
# Voix d'enfants :
#
# Minimale : 250 Hz
# Fondamentale : 400 Hz
# Maximale (y compris les harmoniques) : 20 kHz ou +

def Pass_filter(type, cutoff, audio, sample_rate, order=32):

	if cutoff >= sample_rate / 2:
		cutoff = (sample_rate / 2) - 1

	sos = signal.butter(order // 2, cutoff, btype=type, fs=sample_rate, output='sos')
	filtered_audio = signal.sosfiltfilt(sos, audio, padlen=0, axis=1)

	return filtered_audio

# SRS : Sample Rate Scaling
def Change_sample_rate(audio, way, current_cutoff, target_cutoff):

	if way == "DOWN":
		current_cutoff, target_cutoff = target_cutoff, current_cutoff

	pitched_audio = librosa.resample(audio, orig_sr = current_cutoff * 2, target_sr = target_cutoff * 2, res_type = 'kaiser_best', axis=1)

	return pitched_audio

def Remove_High_freq_Noise(audio, threshold_freq):

	# Calculer la transformée de Fourier
	stft = librosa.stft(audio)
	
	# Calculer la somme des amplitudes pour chaque fréquence dans le spectre
	amplitude_sum = np.sum(np.abs(stft), axis=0)

	# Appliquer un masque pour supprimer les fréquences supérieures lorsque la somme des amplitudes est inférieure au seuil
	stft[:, amplitude_sum > threshold_freq] = 0.0

	# Reconstruire l'audio à partir du STFT modifié
	audio_filtered = librosa.istft(stft)

	return audio_filtered

# Find the high cut-off frequency of the input audio
def Find_Cut_OFF(audio, sample_rate, threshold=0.01):

	# Appliquer un filtre passe-bas pour réduire le bruit
	cutoff_frequency = sample_rate / 2.0  # Fréquence de Nyquist (la moitié du taux d'échantillonnage)

	# Définir l'ordre du filtre passe-bas
	order = 6

	# Calculer les coefficients du filtre passe-bas
	b, a = signal.butter(order, cutoff_frequency - threshold, btype='low', analog=False, fs=sample_rate)

	# Appliquer le filtre au signal audio
	filtered_audio = signal.lfilter(b, a, audio, axis=0)

	# Calculer la FFT du signal audio filtré
	fft_result = np.fft.fft(filtered_audio, axis=0)

	# Calculer les magnitudes du spectre de fréquence
	magnitudes = np.abs(fft_result)

	# Calculer les fréquences correspondant aux bins de la FFT
	frequencies = np.fft.fftfreq(len(audio), 1.0 / sample_rate)

	# Trouver la fréquence de coupure où la magnitude tombe en dessous du seuil
	cut_off_frequencies = frequencies[np.where(magnitudes > threshold)]

	# Trouver la fréquence de coupure maximale parmi toutes les valeurs
	return int(max(cut_off_frequencies))



# - For the code below :
#   MIT License
#
#   Copyright (c) 2023 Anjok07 & aufr33 - Ultimate Vocal Remover (UVR 5)
#
# - https://github.com/Anjok07/ultimatevocalremovergui

MAX_SPEC = 'Max'
MIN_SPEC = 'Min'
AVERAGE  = 'Average'

def Make_Ensemble(algorithm, audio_input):

	if len(audio_input) == 1:  return audio_input[0]
	
	waves = []
	
	if algorithm == AVERAGE:

		waves_shapes = []
		final_waves = []

		for i in range(len(audio_input)):
			wave = audio_input[i]
			waves.append(wave)
			waves_shapes.append(wave.shape[1])

		wave_shapes_index = waves_shapes.index(max(waves_shapes))
		target_shape = waves[wave_shapes_index]
		waves.pop(wave_shapes_index)
		final_waves.append(target_shape)

		for n_array in waves:
			wav_target = to_shape(n_array, target_shape.shape)
			final_waves.append(wav_target)

		waves = sum(final_waves)
		output = waves / len(audio_input)
	else:
		specs = []
		
		for i in range(len(audio_input)):  
			waves.append(audio_input[i])
			
			# wave_to_spectrogram_no_mp
			spec = librosa.stft(audio_input[i], n_fft=6144, hop_length=1024)
			
			if spec.ndim == 1:  spec = np.asfortranarray([spec, spec])

			specs.append(spec)
		
		waves_shapes = [w.shape[1] for w in waves]
		target_shape = waves[waves_shapes.index(max(waves_shapes))]
		
		# spectrogram_to_wave_no_mp
		wave = librosa.istft(ensembling(algorithm, specs), n_fft=6144, hop_length=1024)
	
		if wave.ndim == 1:  wave = np.asfortranarray([wave, wave])

		output = to_shape(wave, target_shape.shape)

	return output

def ensembling(a, specs):
	for i in range(1, len(specs)):
		if i == 1:
			spec = specs[0]

		ln = min([spec.shape[2], specs[i].shape[2]])
		spec = spec[:,:,:ln]
		specs[i] = specs[i][:,:,:ln]
		
		if MIN_SPEC == a:
			spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
		elif MAX_SPEC == a:
			spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

	return spec

def to_shape(x, target_shape):
	padding_list = []
	for x_dim, target_dim in zip(x.shape, target_shape):
		pad_value = (target_dim - x_dim)
		pad_tuple = ((0, pad_value))
		padding_list.append(pad_tuple)
	
	return np.pad(x, tuple(padding_list), mode='constant')
