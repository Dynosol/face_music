import mido
import time
from copy import deepcopy
from random import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
import MIDI_Funcs

# Open MIDI output ports
arpp_out = mido.open_output('Python A 7')
bass_out = mido.open_output('Python B 8')
alto_out = mido.open_output('Python C 9')
outportSets = [arpp_out, bass_out, alto_out]
MIDI_Funcs.niceMidiExit(outportSets)

# Define intervals and settings
scale_set = {
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Minor': [0, 2, 3, 5, 7, 8, 10],
    'HarmMinor': [0, 2, 3, 5, 7, 8, 11],
    'PentMajor': [0, 2, 4, 7, 9],
    'PentMinor': [0, 2, 3, 7, 9],
    'justFifthLol': [0, 7],
    'No.': [0]
}

sevenNotePriority = [0.99, 0.3, 0.8, 0.7, 0.9, 0.3, 0.4]
pentNotePriority = [0.9, 0.7, 0.8, 0.8, 0.7]
scale_arbitraryPriorities = {
    'Major': sevenNotePriority,
    'Minor': sevenNotePriority,
    'HarmMinor': sevenNotePriority,
    'PentMajor': pentNotePriority,
    'PentMinor': pentNotePriority,
    'justFifthLol': [0.9, 0.8],
    'No.': [0.1],
}

noteRangeMin = 24
noteRangeMax = 87

def getNoteSet(baseNote, scaleName='Major'):
    intervalSet = scale_set[scaleName]
    arbitraryIntervalPriority = scale_arbitraryPriorities[scaleName]
    while baseNote >= noteRangeMin:
        baseNote -= 12
    outNotes = []
    outPriority = []
    while baseNote < noteRangeMax:
        for ii in range(len(intervalSet)):
            fooNote = baseNote + intervalSet[ii]
            if fooNote < noteRangeMin:
                continue
            if fooNote > noteRangeMax:
                break
            outNotes.append(fooNote)
            outPriority.append(arbitraryIntervalPriority[ii])
        baseNote += 12
    return np.array(outNotes), np.array(outPriority)

# Chord sequence
chordSequence = [
    [50, 'PentMinor'],  # Dm
    [55, 'Major'],      # G
    [60, 'Major'],      # C
    [57, 'HarmMinor'],  # Am
]

timeDelay = 0.15
scaleMotionVelocity = 0
chordSequenceDuration = 30
currentChordIndex = 0
notesSinceChordChange = 0

currChord, currPriority = getNoteSet(chordSequence[currentChordIndex][0], chordSequence[currentChordIndex][1])
bass_out.send(mido.Message('note_on', note=chordSequence[currentChordIndex][0] - 24))

# Define graphing functions
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})
plt.subplots_adjust(top=0.9, left=0.07, right=0.97, bottom=0.1)

timeVals = np.arange(chordSequenceDuration * 3)

leadHist_notes = np.zeros(chordSequenceDuration * 3, np.int16)
leadHist_velos = np.full(chordSequenceDuration * 3, -2, np.int16)
altoHist_notes = np.zeros(chordSequenceDuration * 3, np.int16)
altoHist_velos = np.full(chordSequenceDuration * 3, -2, np.int16)
bassHist_notes = np.zeros(chordSequenceDuration * 3, np.int16)
bassHist_velos = np.full((chordSequenceDuration * 3), -2, np.int16)
bassHist_notes[0] = chordSequence[currentChordIndex][0] - 24
bassHist_velos[0] = 100

def np_push(np_array, push_val):
    np_array = np.roll(np_array, 1)
    np_array[0] = push_val
    return np_array

def plotHistData(fooTimeVals, notes, velos, color, pltAx):
    velosNorm = np.array(velos, np.float16) / 127
    noteStarts = np.where(velos >= 0)[0]
    if len(noteStarts) > 0:
        pltAx.scatter(fooTimeVals[noteStarts], notes[noteStarts], alpha=velosNorm[noteStarts], c=color)
    ii = 0
    lastNote = 0
    while ii < len(notes):
        if velos[ii] >= 0:
            pltAx.plot(
                [fooTimeVals[ii], fooTimeVals[lastNote]],
                [notes[ii], notes[ii]],
                alpha=velosNorm[ii],
                c=color,
            )
            lastNote = ii
        elif velos[ii] == -2:
            lastNote = ii
        ii += 1

currentNote = currChord[0]
altoNote = currentNote
currentTick = -1

def printSequence(fullPrint=True):
    timeSigTopStr = str(round(chordSequenceDuration / sigDivCount, 3)) if chordSequenceDuration % sigDivCount != 0 else str(round(chordSequenceDuration / sigDivCount))
    if fullPrint:
        print(f"\n Sequence: Pattern: {chordSequenceDuration}/{sigDivCount}     ({timeSigTopStr} beats per measure)")
        print(f"     Part Settings:   altoSetting:{altoSetting.ljust(10)}  bassSetting:{bassSetting}")
    print(f"     altoThresh: {str(round(altoThreshold, 3)).ljust(5, ' ')}   leadLegato: {str(round(leadLegato, 3)).ljust(5, ' ')}", end='')

# Begin loop
while True:
    startTime = time.time()
    currentTick += 1
    notesSinceChordChange += 1
    if notesSinceChordChange >= chordSequenceDuration:
        notesSinceChordChange = 0
        currChord, currPriority = getNoteSet(chordSequence[currentChordIndex][0], chordSequence[currentChordIndex][1])
        bass_out.send(mido.Message('note_off', note=chordSequence[currentChordIndex][0] - 24))
        currentChordIndex = (currentChordIndex + 1) % len(chordSequence)
        bass_out.send(mido.Message('note_on', note=chordSequence[currentChordIndex][0] - 24))

    previousNote = currentNote
    playNoteOdds = random()
    if notesSinceChordChange == 0:
        playNoteOdds *= 20
    elif notesSinceChordChange % 4 == 0:
        playNoteOdds *= 5
    if gapCount > 10:
        playNoteOdds *= gapCount - 8
    elif 0 < gapCount < 4:
        playNoteOdds /= 2
    elif 4 <= gapCount < 8:
        playNoteOdds /= 1.5

    doPlayNote = playNoteOdds > 0.3
    gapCount = 0 if doPlayNote else gapCount + 1

    fooPriority = deepcopy(currPriority)
    for ii in range(len(currChord)):
        fooPriority[ii] *= pow(0.8, abs(currChord[ii] - (scaleMotionVelocity + previousNote)))
        fooPriority[ii] *= 0.5 - abs(len(fooPriority) / 2 - ii) / len(fooPriority)

    prioritySum = sum(fooPriority)
    for ii in range(len(currChord)):
        fooPriority[ii] /= prioritySum

    prioritySumSoFar = 0
    randomValue = random()
    selectedPos = 0
    for ii in range(len(currChord)):
        prioritySumSoFar += fooPriority[ii]
        if randomValue < prioritySumSoFar:
            selectedPos = ii
            break

    if doPlayNote:
        currentNote = currChord[selectedPos]
        noteVel = int(random() * (50 * (currPriority[selectedPos] > 0.5) + 30) + 30)
        arpp_out.send(mido.Message('note_off', note=previousNote))
        arpp_out.send(mido.Message('note_on', note=currentNote, velocity=noteVel))
        scaleMotionVelocity = currentNote - previousNote
    else:
        scaleMotionVelocity /= 1.5

    if random() > 0.75 - 0.5 * (gapCount > 0):
        arpp_out.send(mido.Message('note_off', note=previousNote))

    prevAltoNote = altoNote
    altoNote = currChord[selectedPos]
    noteVelAlto = int(random() * (50 * (currPriority[selectedPos] > 0.5) + 30) + 30)
    alto_out.send(mido.Message('note_off', note=prevAltoNote))
    alto_out.send(mido.Message('note_on', note=altoNote, velocity=noteVelAlto))

    leadHist_notes = np_push(leadHist_notes, currentNote)
    leadHist_velos = np_push(leadHist_velos, noteVel if doPlayNote else -2)
    altoHist_notes = np_push(altoHist_notes, altoNote)
    altoHist_velos = np_push(altoHist_velos, noteVelAlto if doPlayNote else -2)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Note")
    ax[1].set_title("Notes vs Time\nLead (B) Alto (G) Bass (O)")
    ax[1].set_xlim(0, chordSequenceDuration * 3)
    ax[1].set_ylim(12, noteRangeMax + 1)
    plotHistData(timeVals, bassHist_notes, bassHist_velos, 'orange', ax[1])
    plotHistData(timeVals, altoHist_notes, altoHist_velos, 'green', ax[1])
    plotHistData(timeVals, leadHist_notes, leadHist_velos, 'blue', ax[1])

    plt.show()
    plt.pause(timeDelay)
    ax[1].clear()
    endTime = time.time()

    if timeDelay - (endTime - startTime) > 0:
        time.sleep(timeDelay - (endTime - startTime))
    else:
        print("TIMING ERROR!!! Not enough time to process generation between ticks")