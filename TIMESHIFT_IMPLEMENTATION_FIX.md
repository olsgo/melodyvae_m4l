# Timeshift Implementation Fix

This document describes the changes made to fix the timing variation (timeshift) functionality by adopting the proven logic from naotokui/RhythmVAE_M4L.

## Changes Made

### 1. Added BEAT_RESOLUTION Constant

**File**: `src/constants.js`
- Added `BEAT_RESOLUTION = 4.0` constant for consistent timing calculations
- This replaces hardcoded values and matches RhythmVAE's approach

### 2. Updated Timing Calculation Logic

**File**: `melodyvae.js`
- Modified `getNoteIndexAndTimeshift()` function to use `BEAT_RESOLUTION` instead of hardcoded `4.0`
- Updated `encode_add` handler to use consistent timing unit calculation
- This ensures timing calculations are consistent across MIDI parsing and encoding

### 3. Improved Timeshift Loss Coefficient

**File**: `src/vae.js`
- Increased `TIME_LOSS_COEF` from `2.0` to `5.0` to match RhythmVAE
- This gives more weight to timeshift learning during training

### 4. Fixed Timeshift Output Scaling

**File**: `melodyvae.js`
- Updated generation function to use RhythmVAE-style scaling: `-1 to +1` → `0 to 127`
- Applied proper grid_offset scaling after the initial mapping
- This ensures timeshift values are properly scaled for output to Max/MSP

## Technical Details

### Timeshift Range
- **Internal representation**: `-1.0` to `+1.0` (normalized)
  - `-1.0` = maximum early timing
  - `0.0` = exactly on beat
  - `+1.0` = maximum late timing

### Output Scaling
- **Output range**: `0` to `127` (MIDI-compatible)
- **Formula**: `Math.floor(scale(timeshift, -1, 1, 0, 127)) * grid_offset`

### Timing Calculation
- **Unit duration**: `(60.0 / tempo) / BEAT_RESOLUTION`
- **BEAT_RESOLUTION**: `4.0` (16th note subdivisions per quarter note)
- **Centering**: Uses half-unit offset for proper quantization

## Backward Compatibility

All changes maintain complete backward compatibility:
- BEAT_RESOLUTION = 4.0 produces identical results to the previous hardcoded approach
- Existing models and data will work without modification
- Only the timeshift learning and output scaling are improved

## Expected Improvements

1. **Better Timeshift Learning**: Higher loss coefficient improves model's ability to learn timing variations
2. **Consistent Timing**: BEAT_RESOLUTION ensures all timing calculations use the same base unit
3. **Proper Output Scaling**: RhythmVAE-style scaling provides better integration with Max/MSP
4. **Configurable Resolution**: BEAT_RESOLUTION can be adjusted for different timing granularities if needed

## Testing

The implementation has been tested to verify:
- Timeshift calculations produce correct normalized values (-1 to +1 range)
- Output scaling properly maps to 0-127 MIDI range
- Backward compatibility with existing timing logic
- Proper centering and quantization behavior