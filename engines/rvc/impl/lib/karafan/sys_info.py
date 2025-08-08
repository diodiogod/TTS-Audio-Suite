
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, sys, subprocess, json, regex
from distutils import spawn

def Get(font_size):

	import platform, psutil

	system = platform.system()

	html  = '<pre style="font: bold '+ font_size +' monospace; line-height: 1.4">'
	html += "****  System Informations  ****<br><br>"

	# Get the total virtual memory size (in bytes)
	total_virtual_memory = psutil.virtual_memory().total
	unit_index = 0
	units = ['B', 'KB', 'MB', 'GB', 'TB']

	# Convert size into larger units until size is less than 1024
	while total_virtual_memory >= 1024:
		total_virtual_memory /= 1024
		unit_index += 1

	html += "Python : "+ regex.sub(r'\(.*?\)\s*', '', sys.version) +"<br>"
	html += f"OS : {system} { platform.release() }<br>"
	html += f"RAM : {total_virtual_memory:.2f} {units[unit_index]}<br>"
	html += f"Current directory : { os.getcwd() }<br><br>"

	html += "****    CPU Informations    ****<br><br>"
	match system:
		case 'Windows':  # use 'wmic'
			try:
				cpu_info = subprocess.check_output(['wmic', 'cpu', 'get', 'Caption,MaxClockSpeed,NumberOfCores,NumberOfLogicalProcessors', '/FORMAT:CSV'], shell=True, stderr=subprocess.STDOUT).decode('utf-8')
				cpu_info = cpu_info.split('\n')[-2].strip()  # catch the last line
				# Split values
				cpu_info = cpu_info.split(',')
				html += f"CPU : {cpu_info[1]}<br>"
				html += f"Cores : {cpu_info[3]}<br>"
				html += f"Threads : {cpu_info[4]}<br>"
				html += f"MaxClock Speed : {cpu_info[2]} MHz"
			except FileNotFoundError:
				html += "--> Can't get CPU infos : 'wmic' tool is not available on this platform."

		case 'Linux':  # use 'lscpu'
			try:
				cpu_info = subprocess.check_output(['lscpu', '-J'], shell=True, stderr=subprocess.STDOUT).decode('utf-8')
				try:
					cpu_info = json.loads(cpu_info)
					if 'lscpu' in cpu_info:
						sockets = cores = threads = 1
						for item in cpu_info["lscpu"]:
							if 'field' in item and 'data' in item:
								data = item['data']
								match item['field']:
									case "Architecture:":		html += f"Arch : {data}<br>"
									case "Model name:":			html += f"CPU : {data}<br>"
									case "CPU max MHz:":		html += f"MaxClock Speed : {int(data)} MHz<br>"
									case "Socket(s):":			sockets = int(data)
									case "Core(s) per socket:":	cores   = int(data)
									case "Thread(s) per core:":	threads = int(data)
						
						html += f"Cores : {cores * sockets}<br>"
						html += f"Threads : {threads * cores * sockets}"

				except json.decoder.JSONDecodeError:
					html += "--> Can't get CPU infos : 'lscpu' doesn't work correctly on this platform ??"
					
			except FileNotFoundError:
				html += "--> Can't get CPU infos : 'lscpu' tool is not available on this platform."

		case 'Darwin':  # For macOS, use 'sysctl'
			try:
				## TODO : decode CPU infos on macOS
				html += "CPU : " + subprocess.check_output(['sysctl', 'machdep.cpu'], shell=True, stderr=subprocess.STDOUT).decode('utf-8')
			except FileNotFoundError:
				html += "--> Can't get CPU infos : 'sysctl' tool is not available on this platform."

		case _:
			# For other platforms, display a generic message
			html += "--> CPU informations are not available for this platform."

	html += "<br><br>****    GPU Informations    ****<br><br>"
	try:
		if platform.system() == "Windows":
			# If the platform is Windows and nvidia-smi 
			# could not be found from the environment path, 
			# try to find it from system drive with default installation path
			nvidia_smi = spawn.find_executable('nvidia-smi')
			if nvidia_smi is None:
				nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
		else:
			nvidia_smi = "nvidia-smi"

		# Nvidia details information
		gpu_info = subprocess.check_output(nvidia_smi, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
		
		html += '<div style="line-height: 1; ">'+ gpu_info +'</div><br>'

		if gpu_info.find('failed') >= 0:
			html += "<b>GPU runtime is disabled. You can only use your CPU with available RAM.</b>"
		elif gpu_info.find('Tesla T4') >= 0:
			html += "You got a Tesla T4 GPU. (speeds are around  10-25 iterations/sec)"
		elif gpu_info.find('Tesla P4') >= 0:
			html += "You got a Tesla P4 GPU. (speeds are around  8-22 iterations/sec)"
		elif gpu_info.find('Tesla K80') >= 0:
			html += "You got a Tesla K80 GPU. (This is the most common and slowest gpu, speeds are around 2-10 iterations/sec)"
		elif gpu_info.find('Tesla P100') >= 0:
			html += "You got a Tesla P100 GPU. (This is the FASTEST gpu, speeds are around  15-42 iterations/sec)"
	
	except subprocess.CalledProcessError as e:
		print(f"Erreur lors de l'exécution de nvidia-smi : {e.returncode}\n{e.output.decode('utf-8')}")
	except FileNotFoundError:
		html += "<b>--> Can't get GPU infos : 'nvidia-smi' tool is not available on this platform.</b>"

	html += "</pre>"
	
	return html
