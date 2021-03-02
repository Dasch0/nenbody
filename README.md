# nenbody

## Introduction
Nenbody is intended to be a multi-agent 2d simulation for simple control algorithms and computer vision processing. 

## Background
This repository is a continuation of work from both [nenserver](https://github.com/Dasch0/nenserver) and [nenclient](https://github.com/Dasch0/nenclient), originally started in C++ and python. Demos of the work done in those repos are below:

[Acceleration and landing](https://www.youtube.com/watch?v=I9AHZ5UcHEc)
[Error based control](https://www.youtube.com/watch?v=ouFYhsppqWI)
[Neural Control With Inhibitors](https://www.youtube.com/watch?v=WhXmLyYfXIs)


## Demos
[Basic vision with gravity controller](https://youtu.be/SQn7Y11cVWA)

This demo shows single channel vision, scaled from a 1d line of pixels to a 2d image in the viewport. A simple nbody gravity simulation is controlling the movement of each entity.

[Basic vision with placeholder flocking controller](https://youtu.be/infru4_VE_I)

This demo shows single channel vision, scaled from a 1d line of pixels to a 2d image in the viewport. A crude attempt at a flocking algorithm controls the movement of each entity, and maintains a stable set of neighbors in view. Note that this controller has access to the location of each entity, and is not using the visual data.
