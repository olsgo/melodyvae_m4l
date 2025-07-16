# MelodyVAE_M4L
Max for Live(M4L) Melody generator using Variational Autoencoder(VAE) 

A derivative of [RhythmVAE_M4l](https://github.com/naotokui/RhythmVAE_M4L), optimized for Apple Silicon and enhanced with advanced features.

## Features
- **AI-powered melody generation** using Variational Autoencoders
- **Apple Silicon optimization** with TensorFlow.js 4.x
- **Pattern encoding** to encode existing melodies into latent space
- **Model bending** for creative sound exploration with noise injection
- **Improved training performance** with validation loss monitoring
- **Enhanced error handling** and user feedback
- **Real-time generation** with adjustable threshold controls

## How it works

The device uses a Variational Autoencoder (VAE) trained on MIDI melody data to generate new melodic patterns. You can:

1. **Train** the model with your own MIDI files
2. **Generate** new melodies by exploring the 2D latent space
3. **Encode** existing patterns to find their position in latent space
4. **Bend** the model with noise for creative variations

## Requirements
- **Mac**: Ableton Live Suite 10.1.2 or later with Max for Live
- **Windows**: Ableton Live Suite 10.1.2 and standalone Max 8.1.2 or later

*Note: On Windows, you need to set the path to external standalone Max installation in Ableton Live preferences.*

## Installation

### Quick Start
1. Use the pre-built device from the `/release` directory for immediate use

### Development Setup
1. Open the Max project file
2. Install dependencies by running `script npm install` in Max
3. Set Max for Live Device Type to `MIDI` in Project Inspector

**Important Installation Notes:**
- **Path Requirements**: The project must be located in a path without spaces due to `@tensorflow/tfjs-node` build requirements
- **Apple Silicon**: Uses TensorFlow.js 4.x with native Apple Silicon support
- **Node.js**: Requires Node 18+ (Node 20 recommended, included with Max 9)

### Troubleshooting Installation
- If npm install fails, try moving the project to a path without spaces
- For Apple Silicon Macs, ensure you're using the native version of Node.js
- If dependencies fail to install, try clearing npm cache: `npm cache clean --force`

## Usage

### Training the Model
1. **Load MIDI Data**: Click the "Load MIDI" button and select individual MIDI files or a folder
2. **Configure Training**: Set the number of epochs (default: 100)
3. **Start Training**: Click "Train" to begin the training process
4. **Monitor Progress**: Watch training and validation loss in real-time

### Generating Melodies
1. **Set Parameters**: Adjust Z1, Z2 coordinates and threshold values
2. **Add Noise**: Optionally add noise for variation
3. **Generate**: Click "Generate" to create new patterns
4. **Fine-tune**: Adjust threshold min/max to filter note density

### Advanced Features

#### Pattern Encoding
1. **Record Pattern**: Input a melody pattern 
2. **Encode**: Find the pattern's position in latent space
3. **Explore**: Use the encoded coordinates as starting points for generation

#### Model Bending
1. **Add Noise**: Use the "Bend" function with noise range parameter
2. **Creative Exploration**: Bend the model to discover new sonic territories
3. **Reset**: Clear and reload the model to return to original state

## MIDI Mapping

The device processes MIDI notes in the range C3-B4 (MIDI notes 48-71), mapping them to a 24-class chromatic system. Notes are quantized to 16th note resolution over 2-bar loops (32 steps).

## Model Architecture

The VAE consists of:
- **Encoder**: Processes onset, velocity, and duration data through separate dense layers
- **Latent Space**: 2-dimensional space for easy visualization and control
- **Decoder**: Generates onset, velocity, and duration patterns from latent coordinates

### Training Optimizations
- **Batch Size**: 128 for optimal training speed
- **Validation Monitoring**: Real-time validation loss tracking
- **Memory Efficiency**: Optimized data handling for large MIDI datasets
- **Apple Silicon**: Native performance on M1/M2 Macs

## Known Issues
- Folder names with special characters `[]?*!|@` may cause issues
- Grid view is display-only (changes don't affect generated patterns)
- Large MIDI files may require significant training time

## File Structure
```
melodyvae_m4l/
├── src/                    # Source code
│   ├── vae.js             # VAE model implementation
│   ├── utils.js           # Utility functions
│   ├── data.js            # Data handling
│   └── constants.js       # Configuration constants
├── melodyvae.js           # Main application logic
├── melodyvae.maxpat       # Max for Live patch
├── package.json           # Node.js dependencies
└── README.md             # This file
```

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with comprehensive testing
4. Submit a pull request

### Building from Source
1. Clone the repository
2. Run `npm install` (ensure path has no spaces)
3. Open the Max project file
4. Export as Max for Live device

## Technical Details

### Dependencies
- `@tensorflow/tfjs-node`: 4.22.0 (Apple Silicon compatible)
- `@magenta/music`: 1.23.1 (Music processing utilities)
- `@tonejs/midi`: 2.0.7 (MIDI file parsing)

### Performance Considerations
- **Memory Usage**: Scales with MIDI dataset size
- **Training Time**: Depends on data size and epoch count
- **Generation Speed**: Real-time on modern hardware

## License
ISC License - see LICENSE file for details

## Acknowledgments
- Based on [RhythmVAE_M4L](https://github.com/naotokui/RhythmVAE_M4L) by Nao Tokui
- TensorFlow.js team for Apple Silicon optimization
- Max/MSP and Ableton Live communities
