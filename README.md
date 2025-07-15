# MelodyVAE_M4L
Max for Live(M4L) Melody generator using Variational Autoencoder(VAE) 

A derivative of [RhythmVAE_M4l](https://github.com/naotokui/RhythmVAE_M4L)

## Installation

Install the npm dependencies from within Max for Live using `npm install`.

**Important:** the build for `@tensorflow/tfjs-node` fails if the project is
located in a path that contains spaces. Move the `melodyvae_m4l` directory to a
location without spaces (or create a symlink) before running `npm install`.

The project now depends on TensorFlow.js 4.x which includes pre-built
binaries for Apple&nbsp;Silicon. Max 9 ships with Node&nbsp;20 and works
with these versions. If you use your own Node installation, Node&nbsp;18 or
newer is recommended.

Electron is no longer needed for this Max for Live device, so the
outdated `electron` packages were removed from `package.json`.
