import pyfluidsynth
import time

sf2 = '/home/advil/FluidR3_GM/FluidR3_GM.sf2'  # Adjust as needed
midi_file = 'melody_with_chords.mid'

fs = pyfluidsynth.Synth()
fs.start(driver='pulseaudio')  # Try 'alsa' or 'sdl2' if needed
sfid = fs.sfload(sf2)
fs.program_select(0, sfid, 0, 0)

fs.play_midi(midi_file)

# Ensure runtime is long enough for the song
time.sleep(20)  # Replace with song duration

fs.delete()