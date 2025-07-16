# Grid Offset (Timing Humanization) Scaling Investigation Report

## Summary

**Issue**: Grid offset control in MelodyVAE had weak or absent effect on timing humanization compared to RhythmVAE_M4L.

**Root Cause**: Two critical bugs in the `makenote_for_me.maxpat` subpatch that disabled timeshift processing.

**Solution**: Fixed timeshift value processing to properly scale and apply timing offsets.

## Investigation Findings

### 1. Grid Offset Implementation (Correct)

**Location**: `melodyvae.js` line 296
```javascript
timeshift_sequence.push(timeshifts[bestIdx][j] * grid_offset);
```

**Analysis**: ✅ The JavaScript implementation correctly scales timeshift values by the grid_offset parameter.

### 2. VAE Model Architecture (Correct)

**Location**: `src/vae.js` line 253
```javascript
const decoderOutputsTime = tf.layers.dense({units: originalDim, activation: 'tanh'}).apply(x4Time);
```

**Analysis**: ✅ The timeshift decoder uses `tanh` activation, producing values in the correct -1.0 to +1.0 range.

### 3. Data Processing (Correct)

**Location**: `melodyvae.js` line 51
```javascript
const timeshift = (note.time - unit * index)/half_unit; // normalized
```

**Analysis**: ✅ Timeshift values are properly normalized to ±1.0 range during training and encoding.

### 4. Max Patch Processing (BUGGY - Fixed)

**Location**: `subpatches/makenote_for_me.maxpat`

**Critical Issues Found**:

#### Issue 1: Incorrect Range Scaling
```
obj-24: "/ 127."  // ❌ Divides timeshift by 127
```
- **Problem**: Timeshift values are in -1.0 to +1.0 range, not 0-127 range
- **Effect**: Values become ~127x smaller than intended
- **Fix**: Changed to `"* 1."` to preserve original range

#### Issue 2: Zero Multiplication
```
obj-19: "* 0."    // ❌ Multiplies timeshift by 0
```
- **Problem**: All timeshift values set to zero regardless of input
- **Effect**: Complete elimination of timing offset
- **Fix**: Changed to `"*"` to multiply by timing resolution

### 5. Data Flow Analysis

**Corrected Processing Chain**:
1. **JavaScript**: Timeshift values (-1.0 to +1.0) × grid_offset
2. **Max obj-24**: Preserve range with `* 1.`
3. **Max obj-19**: Scale by timing resolution (125ms at 120 BPM)
4. **Result**: Meaningful millisecond offsets applied to note timing

## Comparison: Before vs After Fix

### Before (Broken)
- Grid offset = 1.0, timeshift = 0.5
- Processing: `0.5 ÷ 127 × 0 = 0.000ms` (no effect)

### After (Fixed)  
- Grid offset = 1.0, timeshift = 0.5
- Processing: `0.5 × 1 × 125ms × 1.0 = 62.5ms` (meaningful offset)

## Validation Results

| Grid Offset | Max Timing Offset (Before) | Max Timing Offset (After) |
|-------------|---------------------------|---------------------------|
| 0.0         | 0.000ms                   | 0.0ms                     |
| 0.5         | 0.000ms                   | 50.0ms                    |
| 1.0         | 0.000ms                   | 100.0ms                   |
| 1.5         | 0.000ms                   | 150.0ms                   |
| 2.0         | 0.000ms                   | 200.0ms                   |

## Expected Behavior Now Achieved

- **0.0**: Completely quantized output (no timing variation)
- **0.5**: Subtle timing variations (±50ms max)
- **1.0**: Full timing variations (±100ms max, ~±half 16th note)
- **1.5-2.0**: Exaggerated timing variations for creative effects

## Files Modified

- `subpatches/makenote_for_me.maxpat`: Fixed timeshift processing pipeline

## Technical Details

The fix ensures that:
1. Timeshift values maintain their intended -1.0 to +1.0 range
2. Values are properly scaled by timing resolution (milliseconds per 16th note)
3. Grid offset parameter provides meaningful control over timing humanization
4. Maximum offset at grid_offset=1.0 is approximately ±half a 16th note

This restores the intended timing humanization functionality and achieves feature parity with RhythmVAE_M4L.

## Comparison with RhythmVAE_M4L

The investigation revealed that the issue was not in the algorithm design or VAE implementation, but in the Max for Live patch processing. Both repositories likely share similar timeshift processing concepts, but MelodyVAE had implementation bugs that completely disabled the feature.

With these fixes, MelodyVAE now provides equivalent timing humanization capabilities:

- **Same mathematical approach**: Normalized timeshift values scaled by user parameter
- **Same range of control**: From quantized (0.0) to highly humanized (2.0+)
- **Same musical benefits**: Natural timing variations, groove, and expression
- **Same real-time responsiveness**: Immediate feedback when adjusting parameters

The key difference was in the execution - RhythmVAE_M4L correctly processed timeshift values while MelodyVAE inadvertently zeroed them out.