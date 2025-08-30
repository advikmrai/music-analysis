# import pygame

# # Initialize pygame mixer for MIDI
# pygame.init()
# pygame.mixer.init()
# try:
#     # Load your MIDI file
#     midi_file = 'melody_with_chords.mid'  # Replace with the path to your MIDI file
#     pygame.mixer.music.load(midi_file)
#     pygame.mixer.music.play()
#     print("Playing MIDI file...")
#     # Wait until playback finishes
#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)
#     print("Done.")
# except Exception as e:
#     print(f"Error playing MIDI file: {e}")
# finally:
#     pygame.quit()

import time
import fluidsynth

# Adjust the SoundFont path to where you stored FluidR3_GM.sf2
sf2_path = r'C:\\Users\\sanji\\Downloads\\FluidR3_GM'
midi_file = 'melody_with_chords.mid'

fs = fluidsynth.Synth()
fs.start(driver='dsound')
sfid = fs.sfload(sf2_path)
fs.program_select(0, sfid, 0, 0)
fs.midi_player.add(midi_file)
fs.midi_player.play()
input("Press Enter to stop playback...")
fs.delete()
