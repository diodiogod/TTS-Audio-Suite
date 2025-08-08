
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, configparser

# Default values
Defaults = {
	'AUDIO': {
		'input': "Music",
		'normalize': "-1",
		'output': "Music",
		'output_format': "FLAC",
		'silent': "0",
		'infra_bass': False,
	},
	'PROCESS': {
		'music_1': "----",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "----",
		'bleed_1': "----",
		'bleed_2': "----",
		'high_pass': 14,
		'low_pass': 5,

		'bleed_3': "----",
		'bleed_4': "----",
		'bleed_5': "----",
		'bleed_6': "----",
	},
	'OPTIONS': {
		'speed': "Medium",
		'chunk_size': 500000,
	},
	'BONUS': {
		'DEBUG': True,
		'GOD_MODE': True,
		'KILL_on_END': False,
		'THEME': "",
		'large_gpu': False,
	},
}
Options = {
	'Normalize': [("NONE", "0"), ("-0.5 dB", "-0.5"), ("- 1 dB", "-1"), ("- 2 dB", "-2"), ("- 3 dB", "-3")],
	'Format':	[("FLAC - 24 bits", "FLAC"), ("MP3 - CBR 320K", "MP3"), ("WAV - PCM 16 bits","PCM_16"), ("WAV - FLOAT 32 bits","FLOAT")],
	'Silent':	[("NONE", "0"), ("- 45 dB", "-45"), ("- 50 dB", "-50"), ("- 55 dB", "-55"), ("- 60 dB", "-60")],
	'Speed':	['Fastest', 'Fast', 'Medium', 'Slow', 'Slowest'],
}

Help_Dico = {
	'input':		"- IF « Input » is a folder path, ALL audio files inside this folder will be separated by a Batch processing.<br>- Else, only the selected audio file will be processed.",
	'normalize': 	"Normalize input audio files to avoid <b>clipping</b> and get better results.<br>Normally, <b>you do not have</b> to use this option !!<br>Only for weak or loud songs !",
	'output':		"« Output folder » will be created based on the file's name without extension.<br>For example : if your audio input is named : « 01 - Bohemian Rhapsody<b>.MP3</b> »,<br>then output folder will be named : « 01 - Bohemian Rhapsody »",
	'format':		"Choose your prefered audio format to save audio files.",
	'silent':		"Make silent the parts of audio where dynamic range (RMS) goes below threshold.<br>Don't misundertand : this function is NOT a noise reduction !<br>Its behavior is to clean the final audios from « silent parts » (below -XX dB).",
	'infra_bass':	"This will remove <b>Infra-Bass</b> in your audio files (below 15 Hz).<br>It will leave more place to others frequencies and improve the quality of your audio files.<br>It will also reduce the size of your audio files.",
	'MDX_music':	"Make an Ensemble of extractions with Instrumental selected models.<br>This is used to remove <b>Music</b> before Vocal extractions.",
	'MDX_vocal':	"Make an Ensemble of extractions with Vocals selected models.",
	'MDX_bleed_1':	"Remove <b>Music Bleedings</b> in Vocal extractions.<br><br>DON'T use « <b>Instrum HQ 3</b> » as it catchs too much vocals !!",
	'vocal_pass':	"Set the <b>Passband</b> for <b>Vocal</b> extractions to get more <b>Music</b> back !<br>Defaults : <b>70 Hz</b> (+ slope 15 Hz) & <b>16.5 KHz</b> for this frequencies (Bass - Mid / Harmonics) :<br>Male : 85 - 180 Hz / 14 K , Female : 165 - 255 Hz / 16 K , Children : 250 - 400 Hz / 20 K",
	'MDX_bleed_2':	"Remove <b>Vocal Bleedings</b> in <b>subtracted</b> Music.<br><br>DON'T use « <b>Kim Vocal 2</b> » or « <b>Voc FT</b> » as they remove too much music !!",
	'MDX_bleed_3':	"Remove <b>Music Bleedings</b> in Vocal Bleedings extracted <b>above</b> to get them back in Music !<br>DON'T use « Instrum HQ 3 » as it catchs too much Vocals ! 😉<br>... and : <b>ALL models</b> will carry more or less Vocal bleedings in Music Final !!",
	'speed':		"Fastest : extract in 1 pass with <b>NO</b> SRS (<b>only</b> for Testing)<br>All « Speed » are processed with <b>DENOISE</b> (the same option as in <b>UVR 5</b> 😉)<br>Slowest : is the best quality, but it will take hours to process !! 😝",
	'chunks':		"Chunk size for ONNX models. (default : 500,000)<br><br>Set lower to reduce GPU memory consumption OR <b>if you have GPU memory errors</b> !",
	'debug':		"IF checked, it will save all intermediate audio files to compare in your <b>Audacity</b>.",
	'god_mode':		"Give you the GOD's POWER : each audio file is reloaded IF it was created before,<br>NO NEED to process it again and again !!<br>You'll be warned : You have to <b>delete MANUALLY</b> each file that you want to re-process !",
	'theme':		"Darth Vader or Stormtrooper ?<br><br>Choose your prefered theme for KaraFan !",
	'kill_end':		"On <b>Colab</b> : KaraFan will KILL your session at end of « Processongs », to save your credits !!<br>On <b>your Laptop</b> : KaraFan will KILL your GPU, to save battery (and hot-less) !!<br>On <b>your PC</b> : KaraFan will KILL your GPU, anyway ... maybe it helps ? Try it !!",
#	'large_gpu':	"It will load ALL models in GPU memory for faster processing of MULTIPLE audio files.<br>Requires more GB of free GPU memory.<br>Uncheck it if you have memory troubles.",
}
Presets = [
	{
		'music_1': "MDX23C 8K FFT",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "----",
		'bleed_1': "----",
		'bleed_2': "----",
		'bleed_3': "----",
		'bleed_4': "----",
		'bleed_5': "----",
		'bleed_6': "----",
	},
	{
		'music_1': "----",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "----",
		'bleed_1': "Instrum 3",
		'bleed_2': "Instrum Main",
		'bleed_3': "----",
		'bleed_4': "----",
		'bleed_5': "----",
		'bleed_6': "----",
	},
	{
		'music_1': "----",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "----",
		'bleed_1': "----",
		'bleed_2': "----",
		'bleed_3': "MDX23C 8K FFT",
		'bleed_4': "----",
		'bleed_5': "Instrum 3",
		'bleed_6': "Instrum Main",
	},
	{
		'music_1': "----",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "----",
		'bleed_1': "Instrum 3",
		'bleed_2': "Instrum Main",
		'bleed_3': "MDX23C 8K FFT",
		'bleed_4': "----",
		'bleed_5': "Instrum 3",
		'bleed_6': "Instrum Main",
	},
	{
		'music_1': "----",
		'music_2': "----",
		'vocal_1': "MDX23C 8K FFT",
		'vocal_2': "Voc FT",
		'bleed_1': "Instrum Main",
		'bleed_2': "----",
		'bleed_3': "MDX23C 8K FFT",
		'bleed_4': "----",
		'bleed_5': "----",
		'bleed_6': "----",
	},
]

def Load(Gdrive, isColab):

	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")
	
	if not os.path.isfile(file):
		config = Save(Gdrive, isColab, Defaults)
	else:
		config = configparser.ConfigParser()
		config.optionxform = lambda option: option  # To preserve case of Keys !!
		config.read(file, encoding='utf-8')
		
		# Load default values if not present
		for section in Defaults:
			if section not in config:
				config[section] = {}
			for key in Defaults[section]:
				if key not in config[section]:
					config[section][key] = str(Defaults[section][key])
	
	if  config['AUDIO']['normalize'] not in [x[1] for x in Options['Normalize']]:
		config['AUDIO']['normalize'] = Defaults['AUDIO']['normalize']
	if  config['AUDIO']['output_format'] not in [x[1] for x in Options['Format']]:
		config['AUDIO']['output_format'] = Defaults['AUDIO']['output_format']
	if  config['AUDIO']['silent'] not in [x[1] for x in Options['Silent']]:
		config['AUDIO']['silent'] = Defaults['AUDIO']['silent']
	if  config['OPTIONS']['speed'] not in Options['Speed']:
		config['OPTIONS']['speed'] = Defaults['OPTIONS']['speed']
	
	return config

def Save(Gdrive, isColab, config_dict):
	
	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")

	config = configparser.ConfigParser()
	config.optionxform = lambda option: option  # To preserve case of Keys !!
	config.read_dict(config_dict)

	with open(file, 'w', encoding='utf-8') as config_file:
		config.write(config_file)
	
	return config
