import subprocess

def from_mp3_to_wav(file_name):
    original_format = "./samples/{}".format(file_name)
    new_format = "./samples/{}".format(file_name.replace(".mp3", ".wav"))
    subprocess.call(["ffmpeg", "-i", original_format, new_format])