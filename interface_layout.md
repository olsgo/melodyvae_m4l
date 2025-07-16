# MelodyVAE Max for Live Interface Layout

## Control Panel Layout (Right Side)
```
┌─────────────────────────────┐
│  Live Grid View / Editor    │  
├─────────────────────────────┤
│                             │
│  Pattern Display            │
│                             │
├─────────────────────────────┤
│  Track Selection            │
├─────────────────────────────┤
│  Data Source Menu           │
├─────────────────────────────┤
│  [transpose]    ▲           │  ← Transpose control (-24 to +24)
├─────────────────────────────┤
│  [duration]     ◐           │  ← Duration ratio (0.1 to 12.0)
├─────────────────────────────┤
│  [grid offset]  ◑           │  ← NEW: Grid offset (0.0 to 2.0)
└─────────────────────────────┘
```

## Grid Offset Control Details
- **Position**: Below duration control at y:164
- **Type**: live.dial with float output
- **Range**: 0.0 to 2.0
- **Default**: 1.0 (no scaling)
- **Function**: Scales timing offset values for humanization control

## Parameter Flow
```
Grid Offset Dial → obj-98 → obj-20 inlet 6 → subpatcher inlet 7 → pack obj-2 inlet 5 → generate function
```

This provides real-time control over the timing humanization of generated melodies, addressing the "very quantized" sound issue mentioned in the problem statement.