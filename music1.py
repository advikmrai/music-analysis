from music21 import *

# Gather user parameters
key_input = input("Enter key (e.g., C, D#, F): ")
scale_type = input("Enter scale type (major, minor, pentatonic): ").lower()
num_notes = int(input("How many notes in the melody?: "))
note_length = float(input("Note duration in quarter lengths (e.g., 0.25, 0.5, 1.0): "))

# Create the key and scale objects
user_key = key.Key(key_input)
if scale_type == "major":
    scale_obj = scale.MajorScale(key_input)
elif scale_type == "minor":
    scale_obj = scale.MinorScale(key_input)
elif scale_type == "pentatonic":
    scale_obj = scale.PentatonicScale(key_input)
else:
    scale_obj = scale.MajorScale(key_input)  # default

# Generate melody using the chosen scale
pattern = stream.Stream()
for i in range(num_notes):
    pitch = scale_obj.getPitches(user_key.tonic, user_key.tonic.transpose('P8'))[i % len(scale_obj.getPitches(user_key.tonic, user_key.tonic.transpose('P8')))]
    note_obj = note.Note(pitch)
    note_obj.duration = duration.Duration(note_length)
    pattern.append(note_obj)

# Optionally set an instrument
inst_name = input("Instrument? (e.g., Piano, Violin, Flute): ")
try:
    inst_class = getattr(instrument, inst_name)
    pattern.insert(0, inst_class())
except AttributeError:
    print("Instrument not found, skipping.")

# Save and show
pattern.write('midi', fp='generated_music.mid')
print("Generated music saved as generated_music.mid")
pattern.show('text')
