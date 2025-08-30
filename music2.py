from music21 import *

# User parameters for melody
key_input = input("Enter key (e.g., C, D, F): ")
scale_type = input("Enter scale type (major, minor): ").lower()
num_notes = int(input("How many melody notes?: "))
note_length = float(input("Melody note duration (e.g., 0.5, 1.0): "))

# Choose chord progression
chord_prog = [['C', 'E', 'G'], ['F', 'A', 'C'], ['G', 'B', 'D'], ['C', 'E', 'G']]

# Create key and scale objects
user_key = key.Key(key_input)
if scale_type == 'minor':
    scale_obj = scale.MinorScale(key_input)
else:
    scale_obj = scale.MajorScale(key_input)

melody = stream.Part()
chords = stream.Part()

melody_notes = scale_obj.getPitches(user_key.tonic, user_key.tonic.transpose('P8'))
for i in range(num_notes):
    pitch = melody_notes[i % len(melody_notes)]
    n = note.Note(pitch)
    n.duration = duration.Duration(note_length)
    n.offset = i * note_length
    melody.append(n)
    # Chord at each melody note
    chord_pitches = chord_prog[i % len(chord_prog)]
    c = chord.Chord(chord_pitches)
    c.duration = duration.Duration(note_length)
    c.offset = n.offset
    chords.append(c)

# Optional instruments
melody.insert(0, instrument.Piano())
chords.insert(0, instrument.AcousticGuitar())

# Combine both parts into a score
score = stream.Score([melody, chords])
score.write('midi', fp='melody_with_chords.mid')
print("Saved as melody_with_chords.mid")
score.show('text')
