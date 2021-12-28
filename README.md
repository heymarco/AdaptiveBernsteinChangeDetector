# Hoeffding Change Detector for High Dimensional Data Streams 

Code for the paper "Hoeffding Change Detector for High Dimensional Data Streams"
- wip


## Todos

1. Currently, the implementation is very slow
   1. online updating of mean, std of the windows using Chan's / Welford's algorithm
2. Bernstein works very well, but only if we take the absolute value of epsilon
   1. Investigate, why we can't simply take the difference in the mean between the two subwindows to detect only changes to the upside.
