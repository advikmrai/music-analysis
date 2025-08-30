# Import the athenaCL interpreter
from athenaCL.libATH import athenaCL

# Start athenaCL interpreter session
ath = athenaCL.Interpreter()

# Gather user parameters
print("Enter pitches (comma separated, e.g. c3,e3,g3):")
pitches = input()
print("Choose instrument number (e.g. 3 for sineDrone):")
instrument_num = input()
print("Enter rhythm generator (e.g. rrw for random rhythm):")
rhythm_gen = input()

# athenaCL commands to create PathInstance (PI) and TextureInstance (TI)
ath.cmd(f'pin p1 {pitches}')
ath.cmd(f'tin t1 {instrument_num} {rhythm_gen}')
ath.cmd('timap')   # Displays map of texture/parameters

# Output to MIDI
ath.cmd('eln output.mid')
ath.cmd('elh')
print("Music generated and saved to output.mid")
