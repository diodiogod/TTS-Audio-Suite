
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, gc, requests, subprocess

def Check_dependencies():

	# Dependencies already installed ?
	print("Installing dependencies... This will take few minutes...", end='')
	try:
		subprocess.run(["pip", "install", "-r", "requirements.txt"], text=True, capture_output=True, check=True)
		
		print("\rInstallation done !                                     ") # Clean line
	
	except subprocess.CalledProcessError as e:
		print("Error during Install dependencies :\n" + e.stderr + "\n" + e.stdout + "\n")
		Exit_Notebook()

def Install(params):
	
	# This is setup is only for Colab !!

	if params['isColab'] == False:  return

	Repository  = "https://github.com/Captain-FLAM/KaraFan"
	Version_url = "https://raw.githubusercontent.com/Captain-FLAM/KaraFan/master/App/__init__.py"

	Version = ""; Git_version = ""

	Gdrive = params['Gdrive']
	Project = params['Project']
	DEV_MODE = params['I_AM_A_DEVELOPER']

	if not os.path.exists(Gdrive):
		print("ERROR : Google Drive path is not valid !\n")
		Exit_Notebook()
	
	# Create missing folders
	user_folder = os.path.join(Gdrive, "KaraFan_user")
	os.makedirs(user_folder, exist_ok=True)
	os.makedirs(os.path.join(user_folder, "Models"), exist_ok=True)

	os.chdir(Project)  # For pip install

	Check_dependencies()
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Auto-Magic update !
	try:
		response = requests.get(Version_url)
		if response.status_code == requests.codes.ok:
			Git_version = response.text.split('\n')[0].replace("# Version", "").strip()
		else:
			print("Unable to check version on GitHub ! Maybe you're behind a firewall ?")
	except ValueError as e:
		print("Error processing version data :", e)
	except requests.exceptions.ConnectionError as e:
		print("Connection error while trying to fetch version :", e)

	if Version and Git_version:
		if Git_version > Version:
			print(f'A new version of "KaraFan" is available : {Git_version} !')

			warning = 'You have to download the new version manually from :\n'
			warning += Repository
			warning +='\n... and extract it in your Project folder.\n'
			warning +='Then, you have to "Restart" the notebook to use the new version of "KaraFan" !\n\n'
			
			if DEV_MODE:
				print(warning)
			else:
				if os.path.exists(os.path.join(Project, ".git")):
					try:
						subprocess.run(["git", "-C", Project, "pull"], text=True, capture_output=True, check=True)

						Check_dependencies()

						print('\n\nFOR NOW : you have to go AGAIN in Colab menu, "Runtime > Restart and Run all" to use the new version of "KaraFan" !\n\n')

						Exit_Notebook()
						
					except subprocess.CalledProcessError as e:
						if e.returncode == 127:
							print('WARNING : "Git" is not installed on your system !\n' + warning)
						else:
							print("Error during Update :\n" + e.stderr + "\n" + e.stdout)
							Exit_Notebook()
		else:
			print('"KaraFan" is up to date.')

def Exit_Notebook():
	gc.collect()
	# This trick is copyrigthed by "Captain FLAM" (2023) - MIT License
	# That means you can use it, but you have to keep this comment in your code.
	# After deep researches, I found this trick that nobody found before me !!!
	from google.colab import runtime
	runtime.unassign()
