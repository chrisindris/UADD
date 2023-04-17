## Tools

The shell scripts are the entry points, which set some environment variables and then call launch (passing along the training program, in the form of a shell script).

Launch.py handles the multiple processes, and gets a subprocess for running the training program in that process environment.