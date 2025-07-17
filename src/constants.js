

const MIN_MIDI_NOTE = 48;
const MAX_MIDI_NOTE = 71;

const NUM_MIDI_CLASSES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1; // 24

const LOOP_DURATION = 32; // 2bars x 16th note
const BEAT_RESOLUTION = 4.0; // 16th note resolution (4 subdivisions per quarter note)

const ORIGINAL_DIM = NUM_MIDI_CLASSES * LOOP_DURATION;

const MIN_ONSETS_THRESHOLD = 5; // ignore loops with onsets less than this num

exports.MIN_MIDI_NOTE = MIN_MIDI_NOTE;
exports.MAX_MIDI_NOTE = MAX_MIDI_NOTE;
exports.NUM_MIDI_CLASSES = NUM_MIDI_CLASSES;

exports.LOOP_DURATION = LOOP_DURATION;
exports.BEAT_RESOLUTION = BEAT_RESOLUTION;
exports.ORIGINAL_DIM = ORIGINAL_DIM;

exports.MIN_ONSETS_THRESHOLD = MIN_ONSETS_THRESHOLD;