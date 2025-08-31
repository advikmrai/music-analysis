# #!/home/advil/audio/music-analysis/my-venv/bin/python

# import pyfluidsynth
# import time

# sf2 = '/home/advil/FluidR3_GM/FluidR3_GM.sf2'  # Adjust as needed
# midi_file = 'melody_with_chords.mid'

# fs = pyfluidsynth.Synth()
# fs.start(driver='pulseaudio')  # Try 'alsa' or 'sdl2' if needed
# sfid = fs.sfload(sf2)
# fs.program_select(0, sfid, 0, 0)

# fs.play_midi(midi_file)

# # Ensure runtime is long enough for the song
# time.sleep(20)  # Replace with song duration

# fs.delete()

import pygame
import time

def play_midi_file(midi_file_path):
    """
    Plays a MIDI file using pygame.
    """
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load the MIDI file
        pygame.mixer.music.load(midi_file_path)
        
        # Play the MIDI file
        print(f"Playing {midi_file_path}...")
        pygame.mixer.music.play()
        
        # Wait until the music is finished
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
        print("Playback finished.")
            
    except pygame.error as e:
        print(f"Error playing MIDI file: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{midi_file_path}' was not found.")
    finally:
        # Quit the mixer to free up resources
        pygame.mixer.quit()

if __name__ == "__main__":
    # Specify the path to your MIDI file
    # Replace 'your_midi_file.mid' with the actual path to your file.
    # For example: '/home/your_username/music/my_song.mid'
    file_path = 'melody_with_chords.mid'
    play_midi_file(file_path)

